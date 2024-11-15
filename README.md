## Note - I am not the author of or affiliated with the authors of the 1st place solution. Their original work and github can be found here: https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution

This repository contains an improvement upon the winning solution to the Google - ASL Fingerspelling Recognition competition on kaggle. The original implementation is an improved SqueezeFormer Automatic Speech Recognition architecture. I introduce elements of the new Zipformer model to increase validation test loss and accuracy by approximately 2%. Full training to measure performance of a completely prepared model is TBD.

Competiton website: [link](https://www.kaggle.com/competitions/asl-fingerspelling).\
1st place solution summary: [link](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485). \
Zipformer implementation reference: [link](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer).\
Zipformer paper: [link](https://arxiv.org/html/2310.11230v4).\
Squeezeformer TensorFlow implementation reference: [link](https://github.com/kssteven418/Squeezeformer).\
Squeezeformer PyTorch implementation reference: [link](https://github.com/upskyy/Squeezeformer/).\
Squeezeformer paper: [link](https://arxiv.org/pdf/2206.00888.pdf).\
TFSpeech2TextDecoder uilities from [link](https://github.com/huggingface/transformers/) to support caching and used components related to LLama Attention.\

The solution is an encoder-decoder architecture.



## \[Quoted from the original authors\] Training to reproduce the 1st place solution

In order to reproduce the 1st place solution, two rounds of training are necesseary. In the first round we train a smaller model in order to generate out-of-fold (OOF) predictions which are used as auxiliary target for round two. Finally, model architecture of round two is translated to tensorflow and weights are transfered before we export to a tf-lite model. Note that, for users convinience, we provide the output of step 1 as `datamount/train_folded_oof_supp.csv` so only step 2& 3 would need to be performed to get the final model weights.

      
### 1. Train round 1

Train 4 folds of cfg_1:

```
python train.py -C cfg_1
python train.py -C cfg_1 --fold 1
python train.py -C cfg_1 --fold 2
python train.py -C cfg_1 --fold 3
```

Add oof predictions from step 1. to train_folded.csv and concatenate with supplemental metadata:

```
python scripts/get_train_folded_oof_supp.py 
```

### 2. Train round 2

Train 2x fullfit seeds of cfg_2:

```
python train.py -C cfg_2 --fold -1
python train.py -C cfg_2 --fold -1
```

### 3. TF-Lite conversion

Transfer the resulting weights to a tensorflow equivalent ensemble model and export to tf-lite:

```
python scripts/convert_cfg_2_to_tf_lite.py  
```


The final model is saved under

```
datamount/weights/cfg_2/fold-1/model.tflite 
datamount/weights/cfg_2/fold-1/inference_args.json
```
and can be added to a kaggle kernel and submitted.

      
      
      
