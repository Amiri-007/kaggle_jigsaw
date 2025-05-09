import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class PrimaryCapsule(nn.Module):
    """
    Primary Capsule Layer with dimension=8
    """
    def __init__(self, in_units, num_capsules, in_channels, out_channels, kernel_size=1, stride=1):
        super(PrimaryCapsule, self).__init__()
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, num_capsules * out_channels, kernel_size, stride)
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, in_units]
        batch_size = x.size(0)
        out = self.conv(x)  # [batch_size, num_capsules * out_channels, in_units]
        out = out.view(batch_size, self.num_capsules, self.out_channels, -1)  # [batch_size, num_capsules, out_channels, in_units]
        out = out.permute(0, 1, 3, 2)  # [batch_size, num_capsules, in_units, out_channels]
        
        # Reshape to [batch_size, num_capsules * in_units, out_channels]
        out = out.contiguous().view(batch_size, -1, self.out_channels)
        
        # Squash activation function
        out = self.squash(out)
        
        return out
    
    def squash(self, x, dim=-1, epsilon=1e-8):
        """
        Squash activation function for capsule networks.
        
        Args:
            x: Input tensor
            dim: Dimension along which to squash
            epsilon: Small value to avoid division by zero
            
        Returns:
            Squashed tensor
        """
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm) 
        return scale * x / (torch.sqrt(squared_norm) + epsilon)


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence modeling
    """
    def __init__(self, hidden_dim, attention_dim=None):
        super(SelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = hidden_dim // 2
            
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, hidden_dim]
        # mask shape: [batch_size, seq_len]
        
        # Calculate attention weights
        # [batch_size, seq_len, 1]
        weights = self.projection(x)
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask to float and unsqueeze
            # [batch_size, seq_len, 1]
            mask = mask.float().unsqueeze(-1)
            
            # Set masked positions to large negative value
            weights = weights.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # [batch_size, seq_len, 1]
        weights = F.softmax(weights, dim=1)
        
        # Apply attention weights to input
        # [batch_size, hidden_dim]
        context = torch.sum(weights * x, dim=1)
        
        return context, weights


class LSTMCapsuleNetwork(nn.Module):
    """
    LSTM-Capsule network for text classification
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        capsule_dim: int = 8,
        num_capsules: int = 10,
        attention_dim: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        ema_decay: float = 0.999
    ):
        super(LSTMCapsuleNetwork, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Primary capsule layer
        self.primary_capsule = PrimaryCapsule(
            in_units=lstm_output_dim,
            num_capsules=num_capsules,
            in_channels=lstm_output_dim,
            out_channels=capsule_dim
        )
        
        # Self-attention mechanism
        self.attention = SelfAttention(capsule_dim, attention_dim)
        
        # Output layer
        self.fc = nn.Linear(capsule_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize EMA parameters
        self.ema_decay = ema_decay
        self.ema_params = None
        
    def forward(self, x, lengths=None):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # Create attention mask from lengths
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        else:
            mask = (x != 0)
        
        # Embedding layer
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.dropout(x)
        
        # LSTM layer
        if lengths is not None:
            # Pack padded sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Apply LSTM
            lstm_out, _ = self.lstm(x_packed)
            
            # Unpack sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)
        
        # Transpose for capsule layer
        caps_input = lstm_out.transpose(1, 2)  # [batch_size, hidden_dim * 2, seq_len]
        
        # Primary capsule layer
        capsule_out = self.primary_capsule(caps_input)  # [batch_size, num_capsules * seq_len, capsule_dim]
        
        # Reshape to [batch_size, num_capsules * seq_len, capsule_dim]
        capsule_out = capsule_out.view(batch_size, -1, self.primary_capsule.out_channels)
        
        # Get the correct shape for the capsule attention mask
        # In turbo mode or with smaller inputs, we need to handle shape mismatches
        capsule_seq_len = capsule_out.size(1)
        if capsule_seq_len != mask.view(batch_size, -1).size(1):
            # Recreate mask with correct shape
            if capsule_seq_len == self.primary_capsule.num_capsules * seq_len:
                # Repeat each mask element for each capsule
                capsule_mask = mask.unsqueeze(-1).repeat(1, 1, self.primary_capsule.num_capsules)
                capsule_mask = capsule_mask.view(batch_size, -1)
            else:
                # We need to adapt the mask specifically (use the first part as an approximation)
                # This is a fallback for unusual shape scenarios
                logger.warning(f"Capsule shape mismatch: mask size {mask.size()}, capsule size {capsule_out.size()}")
                capsule_mask = torch.ones(batch_size, capsule_seq_len, device=x.device, dtype=torch.bool)
        else:
            capsule_mask = mask.view(batch_size, -1)
        
        # Apply self-attention
        attended_caps, _ = self.attention(capsule_out, capsule_mask)
        
        # Final prediction
        logits = self.fc(attended_caps)  # [batch_size, 1]
        
        # Apply sigmoid activation
        predictions = torch.sigmoid(logits).squeeze(-1)  # [batch_size]
        
        return predictions
    
    def update_ema_weights(self):
        """
        Update exponential moving average of model weights.
        This helps with model stability and convergence.
        """
        if self.ema_params is None:
            # Initialize EMA parameters
            self.ema_params = {}
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.data.clone()
        else:
            # Update EMA parameters
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = self.ema_params[name] * self.ema_decay + param.data * (1.0 - self.ema_decay)
    
    def apply_ema_weights(self):
        """
        Apply EMA weights to model for inference.
        """
        if self.ema_params is not None:
            # Store current parameters
            current_params = {}
            for name, param in self.named_parameters():
                if param.requires_grad:
                    current_params[name] = param.data.clone()
            
            # Apply EMA parameters
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.ema_params[name])
                    
            return current_params
        return None
    
    def restore_weights(self, weights):
        """
        Restore original weights after using EMA weights.
        
        Args:
            weights: Original weights to restore
        """
        if weights is not None:
            for name, param in self.named_parameters():
                if param.requires_grad and name in weights:
                    param.data.copy_(weights[name])


def create_lstm_capsule_model(config: Dict[str, Any], is_ci_mode: bool = False) -> nn.Module:
    """
    Create LSTM-Capsule model from configuration.
    
    Args:
        config: Model configuration
        is_ci_mode: Whether to use simplified model for CI testing
        
    Returns:
        LSTM-Capsule model
    """
    if is_ci_mode:
        # Create simple model for CI
        model = SimpleLSTMCapsuleNetwork(
            vocab_size=config.get('vocab_size', 50000),
            embedding_dim=config.get('embedding_dim', 8),
            hidden_dim=config.get('hidden_dim', 16),
            dropout=config.get('dropout', 0.2),
            bidirectional=config.get('bidirectional', True)
        )
        return model
    
    # Get model parameters from config
    vocab_size = config.get('vocab_size', 50000)
    embedding_dim = config.get('embedding_dim', 300)
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 2)
    capsule_dim = config.get('capsule_dim', 8)
    num_capsules = config.get('num_capsules', 10)
    dropout = config.get('dropout', 0.2)
    bidirectional = config.get('bidirectional', True)
    freeze_embeddings = config.get('freeze_embeddings', False)
    ema_decay = config.get('ema_decay', 0.999)
    
    # Load pretrained embeddings if provided
    pretrained_embeddings = None
    if config.get('pretrained_embeddings', False) and 'embedding_path' in config:
        from src.utils.text import build_vocab, load_embeddings
        
        # Build vocabulary
        vocab = build_vocab(config.get('vocab_file'))
        
        # Load embeddings
        pretrained_embeddings = load_embeddings(
            config['embedding_path'],
            vocab,
            embedding_dim=embedding_dim
        )
    
    # Create model
    model = LSTMCapsuleNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        capsule_dim=capsule_dim,
        num_capsules=num_capsules,
        dropout=dropout,
        bidirectional=bidirectional,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
        ema_decay=ema_decay
    )
    
    return model


class SimpleLSTMCapsuleNetwork(nn.Module):
    """
    Simplified LSTM-Capsule network for CI testing
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 8,
        hidden_dim: int = 16,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(SimpleLSTMCapsuleNetwork, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Output layer
        self.fc = nn.Linear(lstm_output_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        # x shape: [batch_size, seq_len]
        
        # Embedding layer
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.dropout(x)
        
        # LSTM layer
        if lengths is not None:
            # Pack padded sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Apply LSTM
            lstm_out, _ = self.lstm(x_packed)
            
            # Unpack sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)
        
        # Use last hidden state for classification
        last_hidden = lstm_out[:, -1, :] if lengths is None else lstm_out[torch.arange(lstm_out.size(0)), lengths - 1]
        
        # Apply fully connected layer
        logits = self.fc(last_hidden).squeeze(-1)  # [batch_size]
        
        # Apply sigmoid activation
        predictions = torch.sigmoid(logits)
        
        return predictions 