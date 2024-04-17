#Several SqueezeFormer components where copied/ adapted from https://github.com/upskyy/Squeezeformer/

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
import typing
from torch import Tensor
import math
import numpy as np
import timm
import json
import random
from transformers.models.speech_to_text import Speech2TextConfig, Speech2TextForConditionalGeneration
from transformers.models.speech_to_text.modeling_speech_to_text import shift_tokens_right, Speech2TextDecoder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding
from timm.layers.norm_act import BatchNormAct2d
import copy

from zipscaling import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
    ChunkCausalDepthwiseConv1d,
    Dropout2,
    FloatLike,
    ScheduledFloat,
    Whiten,
    convert_num_channels,
    limit_param_value,
    penalize_abs_values_gt,
    softmax,
    SwooshL,
    SwooshR
)


class Decoder(nn.Module):
    def __init__(self, decoder_config):
        super(Decoder, self).__init__()
        
        self.config = decoder_config
        self.decoder = Speech2TextDecoder(decoder_config) 
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        
        self.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.decoder_end_token_id= decoder_config.eos_token_id
        
    def forward(self,x, labels=None, attention_mask = None, encoder_attention_mask = None):
        
        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
            
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=x, 
                                       attention_mask = attention_mask,
                                       encoder_attention_mask = encoder_attention_mask)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        return lm_logits
            
    def generate(self, x, max_new_tokens=33, encoder_attention_mask=None):

        decoder_input_ids = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.long).fill_(self.decoder_start_token_id)
        for i in range(max_new_tokens-1):  
            decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=x, encoder_attention_mask=encoder_attention_mask)
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            decoder_input_ids = torch.cat([decoder_input_ids,logits.argmax(2)[:,-1:]],dim=1)

            if torch.all((decoder_input_ids==self.decoder_end_token_id).sum(-1) > 0):
                break
                
        return decoder_input_ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        
        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = SwooshL()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

        

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        
        return x


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class DepthwiseConv2d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 2-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = SwooshR()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    # mask_pad = mask.bool().unsqueeze(1)
    def forward(self, x, mask_pad):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        Reference for masking : https://github.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch/wenet/transformer/convolution.py#L26
        """
        # mask batch padding
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        # torch.Size([4, 128, 384])
        x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
        mask_bn = mask_pad.view(-1)
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        '''
        x = self.bn(x)
        '''
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x



def make_scale(encoder_dim):
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None,None,:])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None,None,:])
    return scale, bias


from torch.nn.parameter import Parameter
class BiasNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        # Note: The eps parameter is not used in BiasNorm as per the given definition,
        # but we keep it here for interface compatibility.
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Determine the shape of the bias parameter
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        
        # Initialize parameters similar to LayerNorm for compatibility
        if self.elementwise_affine:
            # As per BiasNorm's definition, the weight here is exp(gamma) - a single scalar
            self.weight = Parameter(torch.tensor([1.0], **(device and {'device': device} or {}),
                                                 **(dtype and {'dtype': dtype} or {})))
            # Bias is a learnable parameter with the same size as the normalized shape
            self.bias = Parameter(torch.zeros(self.normalized_shape, **(device and {'device': device} or {}),
                                              **(dtype and {'dtype': dtype} or {})))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute the RMS value
        rms_val = torch.sqrt(torch.mean((input - self.bias) ** 2, dim=-1, keepdim=True) + self.eps)
        # Scale the input
        output = input / rms_val * self.weight.exp()
        return output

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)




class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        bypass_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        attention_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0
        ),
        conv_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0
        ),
        const_attention_rate: FloatLike = ScheduledFloat(
            (0.0, 0.25), (4000.0, 0.025), default=0
        ),
        ff2_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)
        ),
        ff3_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)
        ),
        bypass_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.5), (4000.0, 0.02), default=0
        )
    ):
        super(SqueezeformerBlock, self).__init__()
        
        '''
        self.mhsa = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,)
        encoder_dim = 144
        num_attention_heads = 4
        attention_dropout_p = 0.1
        self.mhsa_whisper = WhisperAttention(embed_dim = encoder_dim,\
                                       num_heads = num_attention_heads,\
                                       dropout = attention_dropout_p,\
                                       is_decoder = False,\
                                       bias = True)
        '''       

        self.bypass = BypassModule(
            encoder_dim, skip_rate=bypass_skip_rate, straight_through_rate=0
        )
        self.bypass_mid = BypassModule(encoder_dim, straight_through_rate=0)
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)
        self.const_attention_rate = copy.deepcopy(const_attention_rate)
        
        self.self_attn_weights = RotaryEmbeddingWeights(
            LlamaConfig(hidden_size = encoder_dim, 
                        num_attention_heads = num_attention_heads, 
                        max_position_embeddings = 384)
        )

        self.self_attn1 = LlamaAttention(LlamaConfig(hidden_size = encoder_dim, 
                                       num_attention_heads = num_attention_heads, 
                                       max_position_embeddings = 384))

        self.self_attn2 = LlamaAttention(LlamaConfig(hidden_size = encoder_dim, 
                                       num_attention_heads = num_attention_heads, 
                                       max_position_embeddings = 384))
        
        self.feed_forward1 = FeedForwardModule(
            encoder_dim, 3, bypass_dropout_p
        )

        self.feed_forward2 = FeedForwardModule(encoder_dim, 4, bypass_dropout_p)
        self.feed_forward3 = FeedForwardModule(
            encoder_dim, 5, bypass_dropout_p
        )

        self.nonlin_attention = NonlinAttention(
            encoder_dim, hidden_channels=3 * encoder_dim // 4
        )

        self.conv_module1 = ConvModule(
            encoder_dim, conv_kernel_size, conv_expansion_factor, conv_dropout_p
        )
        self.conv_module2 = ConvModule(
            encoder_dim, conv_kernel_size, conv_expansion_factor, conv_dropout_p
        )

        self.norm = BiasNorm(encoder_dim)

        self.balancer1 = Balancer(
            encoder_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.2,
            max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(4.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )    


        self.scale_self_attn_weights, self.bias_self_attn_weights = make_scale(encoder_dim)
        self.scale_self_attn1, self.bias_self_attn1 = make_scale(encoder_dim)
        self.scale_self_attn2, self.bias_self_attn2 = make_scale(encoder_dim)
        self.scale_feed_forward1, self.bias_feed_forward1 = make_scale(encoder_dim)
        self.scale_feed_forward2, self.bias_feed_forward2 = make_scale(encoder_dim)
        self.scale_feed_forward3, self.bias_feed_forward3 = make_scale(encoder_dim)
        self.scale_nonlin_attention, self.bias_nonlin_attention = make_scale(encoder_dim)
        self.scale_conv_module1, self.bias_conv_module1 = make_scale(encoder_dim)
        self.scale_conv_module2, self.bias_conv_module2 = make_scale(encoder_dim)
        self.scale_bypass, self.bias_bypass = make_scale(encoder_dim)
        self.scale_bypass_mid, self.bias_bypass_mid = make_scale(encoder_dim)

        
        '''
        self.mhsa = self.encoder.blocks[0].mhsa
        self.ln_mhsa = self.encoder.blocks[0].ln_mhsa
        self.ln_ff_mhsa = self.encoder.blocks[0].ln_ff_mhsa
        self.ln_conv = self.encoder.blocks[0].ln_conv
        self.ln_ff_conv = self.encoder.blocks[0].ln_ff_conv
        self.ff_mhsa = self.encoder.blocks[0].ff_mhsa
        self.ln_mhsa = self.encoder.blocks[0].ln_mhsa
        self.conv = self.encoder.blocks[0].conv
        self.ff_conv = self.encoder.blocks[0].ff_conv
        '''

    def get_sequence_dropout_mask(
        self, x: Tensor, dropout_rate: float
    ) -> Optional[Tensor]:
        if (
            dropout_rate == 0.0
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask
        


    def forward(self, x, cos, sin, mask):
        mask_pad = ( mask).long().bool().unsqueeze(1)
        mask_pad = ~( mask_pad.permute(0, 2,1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape
        
        residual = x

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = (
                float(self.attention_skip_rate) if self.training else 0.0
            )
        
        x_attn = x * self.scale_self_attn_weights.to(x.dtype) + self.bias_self_attn_weights.to(x.dtype)
        attn_weights = self.self_attn_weights(
            x_attn, cos, sin, attention_mask = mask_pad.unsqueeze(1)
        )

        x = x * self.scale_feed_forward1.to(x.dtype) + self.bias_feed_forward1.to(x.dtype)
        x = x + self.feed_forward1(x)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(
            x, attention_skip_rate
        )

        attn_weights_zip = attn_weights.permute(1, 0, 2, 3)
        selected_attn_weights = attn_weights_zip[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < float(self.const_attention_rate):
            # Make attention weights constant.  The intention is to
            # encourage these modules to do something similar to an
            # averaging-over-time operation.
            # only need the mask, can just use the 1st one and expand later
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(
                selected_attn_weights.dtype
            )
            selected_attn_weights = selected_attn_weights * (
                1.0 / selected_attn_weights.sum(dim=-1, keepdim=True)
            )

        x = x * self.scale_nonlin_attention.to(x.dtype) + self.bias_nonlin_attention.to(x.dtype)
        na = self.nonlin_attention(x, selected_attn_weights.permute(1, 0, 2, 3))
        x = x + (
            na if self_attn_dropout_mask is None else na * self_attn_dropout_mask
        )

        x = x * self.scale_self_attn1.to(x.dtype) + self.bias_self_attn1.to(x.dtype)
        self_attn = self.self_attn1(x, attn_weights)
        x = x + (
            self_attn
            if self_attn_dropout_mask is None
            else self_attn * self_attn_dropout_mask
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0

        x = x * self.scale_conv_module1.to(x.dtype) + self.bias_conv_module1.to(x.dtype)
        x = x + self.sequence_dropout(
            self.conv_module1(
                x, self.conv(x, mask_pad = mask.bool().unsqueeze(1))
            ),
            conv_skip_rate
        )
        
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0

        x = x * self.scale_feed_forward2.to(x.dtype) + self.bias_feed_forward2.to(x.dtype)
        x = x + self.sequence_dropout(
            self.feed_forward2(x), ff2_skip_rate
        )

        x = x * self.scale_bypass_mid.to(x.dtype) + self.bias_bypass_mid.to(x.dtype)
        x = self.bypass_mid(residual, x)
        x = x * self.scale_self_attn2.to(x.dtype) + self.bias_self_attn2.to(x.dtype)
        self_attn = self.self_attn2(x, attn_weights)
        x = x + (
            self_attn
            if self_attn_dropout_mask is None
            else self_attn * self_attn_dropout_mask
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        x = x * self.scale_conv_module2.to(x.dtype) + self.bias_conv_module2.to(x.dtype)
        x = x + self.sequence_dropout(
            self.conv_module2(
                x, self.conv(x, mask_pad = mask.bool().unsqueeze(1))
            ),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        x = x * self.scale_feed_forward3.to(x.dtype) + self.bias_feed_forward3.to(x.dtype)
        x = x + self.sequence_dropout(
            self.balancer_ff3(self.feed_forward3(x)), ff3_skip_rate
        )

        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.norm(x)
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)

        x = x * self.scale_bypass.to(x.dtype) + self.bias_bypass.to(x.dtype)
        x = self.bypass(residual, x)

        x = self.whiten(x)

        return x




class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_index: int = 7,
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """

        for idx, block in enumerate(self.blocks):
            x = block(x, cos, sin, mask)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,out_dim, conv_ch = 3):
        super().__init__()   

        self.in_channels = in_channels = 32 * math.ceil(n_landmarks / 2)
        self.stem_linear = nn.Linear(in_channels,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(conv_ch, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        
    def forward(self, data, mask):


        xc = data.permute(0,3,1,2)
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.permute(0,2,3,1)
        xc = xc.reshape(*data.shape[:2], -1)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc)
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x


class RotaryEmbeddingWeights(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
    
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:, :, :kv_seq_len], sin[:, :, :kv_seq_len])
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)

        past_key_value = (key_states, past_key_value[1]) if use_cache else None
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights.masked_fill_(attention_mask, torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        return attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        '''
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        '''
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (past_key_value[0], value_states) if use_cache else None

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        # self.out_proj = ScaledLinear(
        #     hidden_channels, channels, bias=True, initial_scale=0.05
        # )
        self.out_proj = nn.Linear(hidden_channels, channels, bias=False)

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """.
                Args:
                   x: a Tensor of shape (batch_size, seq_len, embed_dim)
        attn_weights: a Tensor of shape (batch_size, num_heads, seq_len, seq_len)
                Returns:
                   a Tensor with the same shape as x
        """
        x = x.permute(1, 0)
        attn_weights = attn_weights.permute(1, 0, 2, 3)
        """
            NOW:
            x: a Tensor of shape (seq_len, batch_size, embed_dim)
            attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = x * y

        x = self.out_proj(x)
        x = self.whiten2(x)

        x = x.permute(1, 0, 2, 3)
        return x
    



class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(
                self.bypass_scale, min=float(self.scale_min), max=float(self.scale_max)
            )
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = (
                    torch.rand((batch_size, 1), device=ans.device)
                    < straight_through_rate
                )
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig) * bypass_scale









def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.squeeze(1)
    k_embed = k_embed.squeeze(1)
    return q_embed, k_embed



def get_lm_type(lm):
    if 'left_hand' in lm:
        t = 1
    elif 'right_hand' in lm:
        t = 2    
    elif 'face' in lm:
        t = 3  
    elif 'pose' in lm:
        t = 4  
    return t




class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.max_phrase = cfg.max_phrase
        
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        
        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks)//3]])
        self.landmark_types = np.array([get_lm_type(lm) for lm in landmarks])
        self.feature_extractor = FeatureExtractor(n_landmarks=cfg.n_landmarks,out_dim=cfg.encoder_config.encoder_dim)
        self.feature_extractor_lhand = FeatureExtractor(n_landmarks=(self.landmark_types==1).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
        self.feature_extractor_rhand = FeatureExtractor(n_landmarks=(self.landmark_types==2).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
        self.feature_extractor_face = FeatureExtractor(n_landmarks=(self.landmark_types==3).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
        self.feature_extractor_pose = FeatureExtractor(n_landmarks=(self.landmark_types==4).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
       
        rotary_emb = LlamaRotaryEmbedding(cfg.encoder_config.encoder_dim//cfg.encoder_config.num_attention_heads, max_position_embeddings=cfg.max_len)
        self.cos = torch.nn.parameter.Parameter(rotary_emb.cos_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        self.sin = torch.nn.parameter.Parameter(rotary_emb.sin_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)

        self.encoder = SqueezeformerEncoder(
                      input_dim=cfg.encoder_config.input_dim,
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      input_dropout_p= cfg.encoder_config.input_dropout_p,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
        
        self.decoder = Decoder(cfg.transformer_config)
        self.decoder2 = Decoder(cfg.transformer_config) 
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.ce_ignore_index, label_smoothing = cfg.label_smoothing)
        self.aux_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_fc = nn.Linear(cfg.encoder_config.encoder_dim,1)
        self.aux_loss_weight = cfg.aux_loss_weight
        self.return_aux_logits = cfg.return_aux_logits
        self.bwd_loss_weight = cfg.bwd_loss_weight
        
        self.val_mode = cfg.val_mode
        self.decoder_mask_aug = cfg.decoder_mask_aug
        print('n_params:',count_parameters(self))

    def forward(self, batch, debug = False):

        x = batch['input'] # bs, seq_len, n_landmarks, 3
        xp = batch['input'].clone()
        labels = batch['token_ids']
        mask = batch['input_mask'].long()
        label_mask = batch['attention_mask']

        
        # normalise the parts
        for ii in range(4):
            pidx = self.landmark_types==1+ii            
            x_mean = xp[:,:,pidx].mean(2).unsqueeze(2)
            x_std = xp[:,:,pidx].std(2, unbiased=False).unsqueeze(2)
            xp[:,:,pidx] = (xp[:,:,pidx] - x_mean) / x_std
        xp[torch.isnan(xp)] = 0.
        xp[torch.isinf(xp)] = 0.
        dropped_mask = x[:,:,:,:2].sum(-1)==0.
        xp[dropped_mask] = 0.

        x_lhand = self.feature_extractor_lhand(xp[:,:,self.landmark_types==1], mask)
        x_rhand = self.feature_extractor_rhand(xp[:,:,self.landmark_types==2], mask)
        x_face = self.feature_extractor_face(xp[:,:,self.landmark_types==3], mask)
        x_pose = self.feature_extractor_pose(xp[:,:,self.landmark_types==4], mask)
        
        x1 = torch.cat([x_lhand,x_rhand,x_face,x_pose],dim=-1)
        x = self.feature_extractor(x, mask)
        x = x + x1
        x = self.encoder(x, self.cos, self.sin, mask)
        aux_logits = self.aux_fc(x[:,0])
        if debug:
            return x
        
        decoder_labels = labels.clone()
        if self.training:
            m = torch.rand(labels.shape) < self.decoder_mask_aug
            decoder_labels[m] = 62

        logits = self.decoder(x,
                                  labels=decoder_labels, 
                                  encoder_attention_mask=mask.long(),
    #                               attention_mask=label_mask,
                                 )
        
        x_bwd = torch.flip(x, [1])
        mask_bwd = torch.flip(mask, [1])
        lbl_bwd = [dlbl[msk==1] for dlbl,msk in zip(labels.clone(), label_mask)]
        lbl_bwd = [torch.cat((torch.flip(i[:-1], [0]), i[-1:])) for  i in lbl_bwd]
        lbl_bwd = torch.nn.utils.rnn.pad_sequence(lbl_bwd, batch_first=True)
        decoder_lbl_bwd = lbl_bwd.clone()
        if self.training:
            m = torch.rand(lbl_bwd.shape) < self.decoder_mask_aug
            decoder_lbl_bwd[m] = 62
        logits_bwd = self.decoder2(x_bwd,
                                  labels=decoder_lbl_bwd, 
                                  encoder_attention_mask=mask_bwd.long(),
    #                               attention_mask=label_mask,
                                 )
        
        loss_ce = self.loss_fn(logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))   
        loss_ce_bwd = self.loss_fn(logits_bwd.view(-1, self.decoder.config.vocab_size), lbl_bwd.view(-1))   
        loss_aux = self.aux_loss_fn(aux_logits,batch['score'].clamp(0,1)[:,None])
        loss = (1- self.aux_loss_weight) * (loss_ce * (1-self.bwd_loss_weight) + self.bwd_loss_weight * loss_ce_bwd) \
                + self.aux_loss_weight * loss_aux
        
        output = {'loss':loss}
        output['loss_aux'] = loss_aux
        output['loss_ce'] = loss_ce    
        output['loss_ce_bwd'] = loss_ce_bwd     
        
        if not self.training:
            generated_ids_padded = torch.ones((x.shape[0],self.max_phrase), dtype=torch.long, device=x.device) * 59
            
            if self.val_mode == 'padded':
                generated_ids = self.decoder.generate(x,max_new_tokens=self.max_phrase + 1, encoder_attention_mask=mask.long())
                    
            elif self.val_mode == 'cutted':
                generated_ids = torch.ones((x.shape[0],self.max_phrase+1), dtype=torch.long, device=x.device) * 59
                mask_lens = mask.sum(1)
                for lidx in mask_lens.unique():
                    liddx = lidx == mask_lens
                    preds = self.decoder.generate(x[liddx, :lidx],max_new_tokens=self.max_phrase + 1)
                    generated_ids[liddx, :preds.shape[1]] = preds
                    
            cutoffs = (generated_ids==self.decoder.decoder_end_token_id).float().argmax(1).clamp(0,self.max_phrase)
            for i, c in enumerate(cutoffs):
                generated_ids_padded[i,:c] = generated_ids[i,:c]
            output['generated_ids'] = generated_ids_padded
            output['seq_len'] = batch['seq_len']    
            if self.return_aux_logits:
                output['aux_logits'] = aux_logits
        return output
