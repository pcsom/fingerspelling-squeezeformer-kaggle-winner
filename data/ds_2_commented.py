import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
import math

# Move a batch of data to a specified device (e.g., GPU).
def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

# Placeholder variables for custom collate functions, if needed.
tr_collate_fn = None
val_collate_fn = None

# Preprocessing class for data normalization and NaN handling.
class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    # Normalize data to have zero mean and unit variance.
    def normalize(self,x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x
    
    # Replace NaN values with zeros.
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
    # Main processing pipeline applying reshape, normalization, and NaN filling.
    def forward(self, x):
        # Reshape input from (sequence length, 3 * number of landmarks) to (sequence length, number of landmarks, 3).
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1)
        # e.g. sequence is [1, 2, 3, 4, 5, 6] -> [[1, 2], [3, 4], [5, 6]] -> [[1, 3, 5], [2, 4, 6]]
        # Alternate: [x_1, x_2, y_1, y_2, z_1, z_2]
        # Groups original into three subarrays. The inside is then 3 rows, nLandmarks columns:
        # x_1 x_2
        # y_1 y_2
        # z_1 z_2
        # Swapping the last two dimensions is essentially making the transverse of this 2D array, which is:
        # x_1 y_1 z_1
        # x_2 y_2 z_2
        
        x = self.normalize(x)  # Normalize the data.
        x = self.fill_nans(x)  # Fill NaN values with zeros.
        return x

# Data augmentation function to flip data horizontally.
def flip(data, flip_array):
    data[:,:,0] = -data[:,:,0]  # Negate x-coordinates for horizontal flip.
    data = data[:,flip_array]   # Reorder data based on flip array.
    return data

# Interpolate or pad sequences to a fixed length.
def interpolate_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # If data is longer than max_len, crop it.
        data = F.interpolate(data.permute(1,2,0),max_len).permute(2,0,1)
        mask = torch.ones_like(data[:,0,0])
        return data, mask
    
    # If data is shorter, pad it.
    coef = 0  # Padding value.
    padding = torch.ones((diff, data.shape[1], data.shape[2]))
    mask = torch.ones_like(data[:,0,0])
    data = torch.cat([data, padding * coef])
    mask = torch.cat([mask, padding[:,0,0] * coef])
    return data, mask

# Data augmentation function to combine segments from two sequences.
def outer_cutmix(data, phrase, score, data2, phrase2, score2):
    cut_off = np.random.rand()  # Random cut-off point.
    # This cut-off fraction is applied to both the phrase and sequence as an estimation that the selected
    # part of the sequence roughly matches the same proportion of the phrase.
    
    # Calculate cut-off points for phrases and data.
    cut_off_phrase = np.clip(round(len(phrase) * cut_off), 1, len(phrase)-1)
    cut_off_phrase2 = np.clip(round(len(phrase2) * cut_off), 1, len(phrase2)-1)
    cut_off_data = np.clip(round(data.shape[0] * cut_off), 1, data.shape[0]-1)
    cut_off_data2 = np.clip(round(data2.shape[0] * cut_off), 1, data2.shape[0]-1)

    # Combine segments from both data and phrases based on cut-off points.
    if np.random.rand() < 0.5:
        new_phrase = phrase2[cut_off_phrase2:] + phrase[:cut_off_phrase]
        new_data = torch.cat([data2[cut_off_data2:], data[:cut_off_data]])
        new_score = cut_off*score + (1-cut_off) * score2
    else:
        new_phrase = phrase[cut_off_phrase:] + phrase2[:cut_off_phrase2]
        new_data = torch.cat([data[cut_off_data:], data2[:cut_off_data2]])
        new_score = cut_off*score2 + (1-cut_off) * score
    return new_data, new_phrase, new_score

# Define CustomDataset class for loading and processing dataset
class CustomDataset(Dataset):
    # Initialize the dataset with dataframe, configuration, optional augmentation, and mode
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.cfg = cfg  # Store the configuration
        self.df = df.copy()  # Make a copy of the input dataframe
        self.mode = mode  # Training or testing mode
        self.aug = aug  # Optional augmentation function

        # Filter out sequences shorter than the minimum sequence length in training mode
        if mode == 'train':
            to_drop = self.df['seq_len'] < cfg.min_seq_len
            self.df = self.df[~to_drop].copy()
            print(f'New shape {self.df.shape[0]}, dropped {to_drop.sum()} sequences shorter than min_seq_len {cfg.min_seq_len}')

        # Ensure 'score' column exists, defaulting to 1.0, and clip values to [0,1]
        if 'score' not in self.df.columns:
            self.df['score'] = 1.
        self.df['score'] = self.df['score'].clip(0, 1)

        # Load selected columns for landmarks from a configuration file
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks) // 3]])
        # Col names (xyz_landmarks) are x_right_hand_1, y_face_5, etc. But data has each corresponding x/y/z landark group already together 
        # in a tuple of 3 (as x1/y1/z1, x2/y2/z2, etc). So, get rid of x_, y_, z_ prefixes and just keep the landmark identifier (right_hand_1, etc)

        # Identify symmetric landmarks for data flipping
        symmetry = pd.read_csv(cfg.symmetry_fp).set_index('id')
        flipped_landmarks = symmetry.loc[landmarks]['corresponding_id'].values
        self.flip_array = np.where(landmarks[:, None] == flipped_landmarks[None, :])[1]

        self.max_len = cfg.max_len  # Maximum sequence length
        self.processor = Preprocessing()  # Preprocessing module instance

        # Tokenization parameters
        self.max_phrase = cfg.max_phrase
        self.char_to_num, self.num_to_char, _ = cfg.tokenizer
        self.pad_token_id = self.char_to_num[cfg.pad_token]
        self.start_token_id = self.char_to_num[cfg.start_token]
        self.end_token_id = self.char_to_num[cfg.end_token]

        # Augmentation probabilities
        self.flip_aug = cfg.flip_aug
        self.outer_cutmix_aug = cfg.outer_cutmix_aug

        # Data folder path
        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder

        # Prepare phrases for training/testing
        self.df['phrase'] = self.df['phrase'].astype(str)
        if mode == 'train':
            # Separate dataframes for supervised and non-supervised data
            self.supp_df = self.df[self.df['is_sup'] == 1].copy()       #This is a filtering expression; only get rows that have is_sup column val as 1
            self.non_supp_df = self.df[self.df['is_sup'] == 0].copy()
            self.df_gr = self.supp_df.groupby('phrase')
            self.phrases = np.concatenate([self.non_supp_df['phrase'].values, self.supp_df['phrase'].unique()])
        else:
            self.df = self.df[self.df['is_sup'] == 0].copy()
            self.phrases = self.df['phrase'].values

    # Return the length of the dataset
    def __len__(self):
        return len(self.phrases)

    # Retrieve a data item and its corresponding label
    def __getitem__(self, idx):
        # Select data based on mode
        if self.mode == 'train':
            phrase = self.phrases[idx]
            if idx < self.non_supp_df.shape[0]:
                row = self.non_supp_df.iloc[idx]
            else:
                g = self.df_gr.get_group(phrase)
                row = g.sample(1).iloc[0]
        else:
            row = self.df.iloc[idx]
        file_id, sequence_id, phrase, score = row[['file_id', 'sequence_id', 'phrase', 'score']]
        #This works on df; it just returns the values at the 4 columns provided.

        # Load and process data
        data = self.load_one(file_id, sequence_id)
        seq_len = data.shape[0]
        data = torch.from_numpy(data)

        if self.mode == 'train':
            # Apply preprocessing and optional augmentations
            data = self.processor(data)
            if np.random.rand() < self.flip_aug:
                data = flip(data, self.flip_array)
            if np.random.rand() < self.outer_cutmix_aug:
                # Apply outer_cutmix augmentation if condition is met
                participant_id = row['participant_id']
                sequence_id = row['sequence_id']
                mask = (self.df['participant_id'] == participant_id) & (self.df['sequence_id'] != sequence_id)  # Check for sequences by the same participant but different IDs

                # Choose an alternative sequence for cutmix augmentation
                if mask.sum() > 0:
                    row2 = self.df[mask].sample(1).iloc[0]
                    file_id2, sequence_id2, phrase2, score2 = row2[['file_id', 'sequence_id', 'phrase', 'score']]
                    data2 = self.load_one(file_id2, sequence_id2)
                    data2 = torch.from_numpy(data2)

                    # Preprocess the second sequence and combine
                    data2 = self.processor(data2)
                    data, phrase, score = outer_cutmix(data, phrase, score, data2, phrase2, score2)

            # Apply additional augmentations if defined
            if self.aug:
                data = self.augment(data)
        else:
            # For testing mode, only preprocessing is applied
            data = self.processor(data)

        # Adjust data length and generate masks for sequence length management
        data, mask = interpolate_or_pad(data, max_len=self.max_len)

        # Convert phrases to token IDs and generate attention masks
        token_ids, attention_mask = self.tokenize(phrase)

        # Package the processed data and additional information into a feature dictionary
        feature_dict = {
            'input': data,
            'input_mask': mask,
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'seq_len': torch.tensor(seq_len),
            'score': torch.tensor(score),
        }
        return feature_dict

    # Optionally apply external augmentation to the data
    def augment(self, x):
        x_aug = self.aug(image=x)['image']
        return x_aug

    # Convert a phrase to a sequence of token IDs, applying padding and special tokens as necessary
    def tokenize(self, phrase):
        # Convert each character in the phrase to its corresponding token ID
        phrase_ids = [self.char_to_num[char] for char in phrase]
        # Truncate or pad the phrase to ensure it fits within the max phrase length
        if len(phrase_ids) > self.max_phrase - 1:
            phrase_ids = phrase_ids[:self.max_phrase - 1]
        phrase_ids += [self.end_token_id]  # Append end token
        attention_mask = [1] * len(phrase_ids)  # Generate attention mask for actual tokens

        # Pad the tokenized phrase to the max length with pad tokens
        to_pad = self.max_phrase - len(phrase_ids)
        phrase_ids += [self.pad_token_id] * to_pad
        attention_mask += [0] * to_pad  # Extend attention mask for padding

        return torch.tensor(phrase_ids).long(), torch.tensor(attention_mask).long()

    # Set up tokenizer mappings from characters to tokens and vice versa
    def setup_tokenizer(self):
        # Load character to token mappings from configuration
        with open(self.cfg.character_to_prediction_index_fn, "r") as f:
            char_to_num = json.load(f)

        # Extend mappings with special tokens for padding, start, and end of phrases
        n = len(char_to_num)
        char_to_num[self.cfg.pad_token] = n
        char_to_num[self.cfg.start_token] = n + 1
        char_to_num[self.cfg.end_token] = n + 2
        num_to_char = {j: i for i, j in char_to_num.items()}  # Inverse mapping for token to character conversion
        return char_to_num, num_to_char

    # Load a single sequence of landmarks from a file
    def load_one(self, file_id, sequence_id):
        # Construct file path and load data as a numpy array
        path = self.data_folder + f'{file_id}/{sequence_id}.npy'
        data = np.load(path)  # Data shape: sequence length, 3 times number of landmarks
        return data



