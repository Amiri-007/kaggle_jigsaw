import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from .sampling import apply_negative_downsampling, get_sample_weights
import pathlib

class ToxicDataset(Dataset):
    """
    Dataset for toxic comment classification with identity attributes
    """
    def __init__(
        self,
        data_frame: pd.DataFrame,
        tokenizer=None,
        max_length: int = 256,
        is_training: bool = True,
        text_col: str = 'comment_text',
        target_col: str = 'target',
        identity_cols: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.data = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.text_col = text_col
        self.target_col = target_col
        self.cache_dir = cache_dir
        
        # Set default identity columns if none provided
        self.identity_cols = identity_cols or [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness',
            'asian', 'hindu', 'buddhist', 'atheist', 'bisexual', 'transgender'
        ]
        
        # Make sure all identity columns exist in the dataframe
        for col in self.identity_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # ----------------------------------------------------
        # FAST PATH: use pre-tokenised .pt tensor if present
        # ----------------------------------------------------
        tok_cache = getattr(self, "token_cache", None)
        if tok_cache is None and self.cache_dir:
            cache_name = f"{self.tokenizer.name_or_path.replace('/','_')}_{self.max_length}.pt"
            cache_path = pathlib.Path(self.cache_dir) / cache_name
            if cache_path.exists():
                self.token_cache = torch.load(cache_path, map_location="cpu")
                tok_cache = self.token_cache
                print(f"[DataLoader] Loaded token cache {cache_path}")
        if tok_cache is not None:
            input_ids = tok_cache["ids"][idx]
            attn_mask = tok_cache["attn"][idx]
        else:
            text = row[self.text_col]
            if not isinstance(text, str):
                text = "" if pd.isna(text) else str(text)
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids, attn_mask = enc["input_ids"][0], enc["attention_mask"][0]
        
        # Get identity column values
        identity_values = []
        for col in self.identity_cols:
            identity_values.append(float(row[col]))
        
        # Calculate identity sum
        identity_sum = sum(identity_values)
            
        # Get target
        if self.target_col in self.data.columns:
            target = float(row[self.target_col])
        else:
            target = 0.0  # Default for test data
            
        # Tokenize text if tokenizer provided
        if self.tokenizer:
            # Create encoding dictionary
            encoding = {
                'input_ids': input_ids,
                'attention_mask': attn_mask
            }
                
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'target': torch.tensor(target, dtype=torch.float),
                'identity_values': torch.tensor(identity_values, dtype=torch.float),
                'identity_sum': torch.tensor(identity_sum, dtype=torch.float),
                'idx': torch.tensor(idx, dtype=torch.long),
            }
        else:
            # Return text and labels without tokenization
            return {
                'text': text,
                'target': torch.tensor(target, dtype=torch.float),
                'identity_values': torch.tensor(identity_values, dtype=torch.float),
                'identity_sum': torch.tensor(identity_sum, dtype=torch.float),
                'idx': torch.tensor(idx, dtype=torch.long),
            }


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    text_col: str = 'comment_text',
    target_col: str = 'target',
    identity_cols: Optional[List[str]] = None,
    apply_downsampling: bool = True,
    apply_weights: bool = True,
    annotator_weight: bool = False,
    num_workers: int = 4,
    first_epoch: bool = True,
    random_state: int = 42,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Create train and validation dataloaders with optional negative downsampling
    and sample weighting.
    
    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe
        tokenizer: Tokenizer to use for text preprocessing
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        text_col: Column name for text data
        target_col: Column name for target labels
        identity_cols: List of identity column names
        apply_downsampling: Whether to apply negative downsampling (first epoch)
        apply_weights: Whether to apply sample weights for weighted loss
        annotator_weight: Whether to apply weight based on annotator count
        num_workers: Number of workers for dataloader
        first_epoch: Whether this is the first epoch (controls downsampling strategy)
        random_state: Random seed for reproducibility
        cache_dir: Directory to cache tokenized data
        
    Returns:
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        sample_weights: Optional tensor of sample weights for training
    """
    # Apply negative downsampling if requested
    if apply_downsampling:
        train_df = apply_negative_downsampling(
            train_df, 
            target_col=target_col,
            identity_cols=identity_cols,
            first_epoch=first_epoch,
            random_state=random_state
        )
    
    # Create datasets
    train_dataset = ToxicDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=True,
        text_col=text_col,
        target_col=target_col,
        identity_cols=identity_cols,
        cache_dir=cache_dir
    )
    
    valid_dataset = ToxicDataset(
        valid_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=False,
        text_col=text_col,
        target_col=target_col,
        identity_cols=identity_cols,
        cache_dir=cache_dir
    )
    
    # Calculate sample weights if requested
    sample_weights = None
    if apply_weights:
        sample_weights = get_sample_weights(
            train_df, 
            target_col=target_col,
            identity_cols=identity_cols,
            annotator_weight=annotator_weight
        )
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, sample_weights 