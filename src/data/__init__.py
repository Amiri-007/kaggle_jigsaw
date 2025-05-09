from .loaders import ToxicDataset, create_dataloaders
from .sampling import apply_negative_downsampling, get_sample_weights
from .utils import list_identity_columns
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_train_valid(data_path: str, valid_frac: float = 0.05,
                     random_state: int = 1234,
                     sample_frac: float | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    train_df, valid_df = train_test_split(
        df, test_size=valid_frac, random_state=random_state, stratify=(df["target"] >= 0.5)
    )
    return train_df, valid_df

__all__ = [
    'ToxicDataset',
    'create_dataloaders',
    'apply_negative_downsampling',
    'get_sample_weights',
    'list_identity_columns',
    'load_train_valid'
] 