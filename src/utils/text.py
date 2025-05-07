import os
import re
import string
import unicodedata
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Union, Set, Tuple

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace, control characters, etc.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with special token
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Replace email addresses with special token
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Replace user mentions with special token
    text = re.sub(r'@\S+', '[USER]', text)
    
    # Replace numbers with special token
    text = re.sub(r'\d+', '[NUM]', text)
    
    # Replace multiple punctuation with single
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into a list of tokens.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # Normalize text
    text = normalize_text(text)
    
    # Split on whitespace
    tokens = text.split()
    
    # Simple word-level tokenization
    result = []
    for token in tokens:
        # Handle punctuation
        if token in string.punctuation:
            result.append(token)
        else:
            # Check if token ends with punctuation
            if token[-1] in string.punctuation and len(token) > 1:
                result.append(token[:-1])
                result.append(token[-1])
            else:
                result.append(token)
    
    return result

def build_vocab(vocab_file: Optional[str] = None, texts: Optional[List[str]] = None, min_freq: int = 5, max_size: int = 50000) -> Dict[str, int]:
    """
    Build vocabulary from texts or load from file.
    
    Args:
        vocab_file: Path to vocabulary file
        texts: List of texts to build vocabulary from
        min_freq: Minimum frequency for a token to be included
        max_size: Maximum vocabulary size
        
    Returns:
        Dictionary mapping tokens to indices
    """
    # Load vocabulary from file if provided
    if vocab_file and os.path.exists(vocab_file):
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.strip()
                vocab[token] = i
        
        return vocab
    
    # Build vocabulary from texts
    if texts:
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(tokenize(text))
        
        # Count tokens
        counter = Counter(all_tokens)
        
        # Filter by frequency and limit size
        tokens = [token for token, count in counter.most_common(max_size) if count >= min_freq]
        
        # Add special tokens
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[URL]', '[EMAIL]', '[USER]', '[NUM]']
        tokens = special_tokens + [token for token in tokens if token not in special_tokens]
        
        # Create vocabulary
        vocab = {token: i for i, token in enumerate(tokens)}
        
        return vocab
    
    # Return default vocabulary if neither file nor texts provided
    return {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, '[URL]': 5, '[EMAIL]': 6, '[USER]': 7, '[NUM]': 8}

class BasicTokenizer:
    """
    Basic tokenizer for LSTM-based models.
    """
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {i: token for token, i in vocab.items()}
        self.pad_token_id = vocab.get('[PAD]', 0)
        self.unk_token_id = vocab.get('[UNK]', 1)
        self.cls_token_id = vocab.get('[CLS]', 2)
        self.sep_token_id = vocab.get('[SEP]', 3)
    
    def encode(self, text: str, max_length: int = 256, padding: bool = True, truncation: bool = True) -> List[int]:
        """
        Encode text to token ids.
        
        Args:
            text: Text to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequence to max_length
            truncation: Whether to truncate sequence to max_length
            
        Returns:
            List of token ids
        """
        # Tokenize text
        tokens = tokenize(text)
        
        # Add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Truncate if needed
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['[SEP]']
        
        # Convert to ids
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Pad if needed
        if padding and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token ids to text.
        
        Args:
            ids: List of token ids
            
        Returns:
            Decoded text
        """
        # Convert ids to tokens
        tokens = [self.id_to_token.get(id, '[UNK]') for id in ids]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in ['[PAD]', '[CLS]', '[SEP]']]
        
        # Join tokens
        text = ' '.join(tokens)
        
        return text

def load_embeddings(embedding_path: str, vocab: Dict[str, int], embedding_dim: int = 300) -> np.ndarray:
    """
    Load pre-trained embeddings for vocabulary.
    
    Args:
        embedding_path: Path to embedding file
        vocab: Vocabulary to load embeddings for
        embedding_dim: Embedding dimension
        
    Returns:
        Embedding matrix with shape (vocab_size, embedding_dim)
    """
    # Initialize embedding matrix
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
    
    # Set padding token embedding to zeros
    if '[PAD]' in vocab:
        embedding_matrix[vocab['[PAD]']] = np.zeros(embedding_dim)
    
    # Load pre-trained embeddings
    if os.path.exists(embedding_path):
        print(f"Loading embeddings from {embedding_path}")
        
        num_loaded = 0
        with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) < embedding_dim + 1:
                    continue
                
                word = parts[0]
                if word in vocab:
                    embedding_matrix[vocab[word]] = np.array([float(val) for val in parts[1:embedding_dim+1]])
                    num_loaded += 1
        
        print(f"Loaded {num_loaded}/{vocab_size} word embeddings")
    
    return embedding_matrix 