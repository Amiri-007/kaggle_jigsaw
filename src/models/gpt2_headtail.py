import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Model, GPT2PreTrainedModel, GPT2Config, AutoModel
from typing import Optional, Tuple, Dict, Any, List, Union

class GPT2HeadTailClassifier(GPT2PreTrainedModel):
    """GPT-2 model for classification using both head and tail of the input sequence.
    
    This model concatenates the first 128 tokens and last 128 tokens of the input text
    to handle longer sequences more efficiently.
    """
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.num_labels = 1  # Binary classification for toxicity
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # Double the classifier input size to accommodate both head and tail embeddings
        self.classifier = nn.Linear(config.n_embd * 2, self.num_labels)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        head_input_ids: Optional[torch.Tensor] = None,
        head_attention_mask: Optional[torch.Tensor] = None,
        tail_input_ids: Optional[torch.Tensor] = None,
        tail_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_position_ids: Optional[torch.Tensor] = None,
        tail_position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Process head portion through GPT-2
        head_outputs = self.transformer(
            input_ids=head_input_ids,
            attention_mask=head_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=head_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Process tail portion through GPT-2
        tail_outputs = self.transformer(
            input_ids=tail_input_ids,
            attention_mask=tail_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=tail_position_ids,
            past_key_values=None,  # Don't use cache from head
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Get the last hidden states for classification
        # For GPT-2, we typically use the last token's representation for classification
        # Here we'll get the last non-padding token for both head and tail
        
        # Get sequence lengths from attention masks
        head_seq_lengths = head_attention_mask.sum(dim=1) - 1
        tail_seq_lengths = tail_attention_mask.sum(dim=1) - 1
        
        batch_size = head_input_ids.size(0)
        
        # Extract features from the last token of each sequence
        head_features = []
        for i in range(batch_size):
            # Get the last non-padding token's hidden state
            idx = head_seq_lengths[i]
            head_features.append(head_outputs.last_hidden_state[i, idx])
            
        head_features = torch.stack(head_features)
        
        # Same for tail
        tail_features = []
        for i in range(batch_size):
            idx = tail_seq_lengths[i]
            tail_features.append(tail_outputs.last_hidden_state[i, idx])
            
        tail_features = torch.stack(tail_features)
        
        # Concatenate head and tail representations
        pooled_output = torch.cat([head_features, tail_features], dim=1)
        pooled_output = self.dropout(pooled_output)
        
        # Pass through classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
        
        if not return_dict:
            output = (logits,) + head_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": head_outputs.past_key_values,
            "hidden_states": head_outputs.hidden_states,
            "attentions": head_outputs.attentions,
        }


class GPT2HeadTailForSequenceClassification(nn.Module):
    """Wrapper class for GPT2HeadTailClassifier to simplify usage."""
    
    def __init__(self, model_name: str = "gpt2", num_labels: int = 1):
        super().__init__()
        self.config = GPT2Config.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.model = GPT2HeadTailClassifier.from_pretrained(model_name, config=self.config)
        
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def prepare_head_tail_inputs(self, texts, tokenizer, max_length=128):
        """
        Tokenize input texts, extracting the first and last parts of each text
        to handle longer sequences effectively.
        
        Args:
            texts: List of input texts to tokenize
            tokenizer: HuggingFace tokenizer
            max_length: Maximum length for each head and tail sequence
            
        Returns:
            Dictionary of input tensors for the model
        """
        # Tokenize complete texts to get full token IDs
        full_encodings = tokenizer(texts, add_special_tokens=False)
        
        # Prepare head inputs
        head_inputs = tokenizer(
            texts, 
            max_length=max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # For tail, process each text separately to extract the last tokens
        tail_input_ids = []
        tail_attention_masks = []
        
        for i, text in enumerate(texts):
            # Get full token IDs for this text
            full_ids = full_encodings["input_ids"][i]
            
            if len(full_ids) <= max_length:
                # If text is short, use the same tokens but with different positional encoding
                tail_ids = full_ids
            else:
                # Extract last max_length tokens (for GPT-2, we don't need special start/end tokens)
                tail_ids = full_ids[-max_length:]
            
            # Pad to max_length if needed
            padding_length = max_length - len(tail_ids)
            if padding_length > 0:
                tail_ids = [tokenizer.pad_token_id] * padding_length + tail_ids
            
            # Create attention mask (0 for padding, 1 for content)
            tail_mask = [0] * padding_length + [1] * (max_length - padding_length)
            
            tail_input_ids.append(tail_ids)
            tail_attention_masks.append(tail_mask)
        
        # Convert to tensors
        model_inputs = {
            "head_input_ids": head_inputs["input_ids"],
            "head_attention_mask": head_inputs["attention_mask"],
            "tail_input_ids": torch.tensor(tail_input_ids),
            "tail_attention_mask": torch.tensor(tail_attention_masks),
        }
        
        return model_inputs 

class SimpleGPT2HeadTailClassifier(nn.Module):
    """
    Simplified GPT-2 classifier for CI testing.
    Uses a small pre-trained model and freezes weights.
    """
    def __init__(self, pretrained_model_name: str = "distilgpt2"):
        super().__init__()
        self.gpt2 = AutoModel.from_pretrained(pretrained_model_name)
        
        # Freeze GPT-2 parameters for faster training in CI
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        # Simple classifier head
        self.classifier = nn.Linear(self.gpt2.config.n_embd, 1)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_input_ids: Optional[torch.Tensor] = None,
        head_attention_mask: Optional[torch.Tensor] = None,
        tail_input_ids: Optional[torch.Tensor] = None,
        tail_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Use either the standard input_ids or head_input_ids based on what's provided
        if input_ids is not None:
            ids = input_ids
            mask = attention_mask
        else:
            ids = head_input_ids
            mask = head_attention_mask
        
        # Get GPT-2 outputs
        outputs = self.gpt2(
            input_ids=ids,
            attention_mask=mask,
            return_dict=True
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Get sequence lengths from attention masks
        seq_lengths = mask.sum(dim=1) - 1
        batch_size = ids.size(0)
        
        # Extract features from the last token of each sequence
        features = []
        for i in range(batch_size):
            # Get the last non-padding token's hidden state
            idx = seq_lengths[i] if seq_lengths[i] >= 0 else 0
            features.append(last_hidden_state[i, idx])
            
        features = torch.stack(features)
        
        # Get logits
        logits = self.classifier(features)
        
        # Apply sigmoid for probability output
        probs = torch.sigmoid(logits)
        
        return {
            "logits": logits.squeeze(-1),
            "probs": probs.squeeze(-1)
        }

class SimpleGPT2HeadTailForSequenceClassification(nn.Module):
    """Wrapper class for SimpleGPT2HeadTailClassifier to simplify usage."""
    
    def __init__(self, model_name: str = "distilgpt2", num_labels: int = 1):
        super().__init__()
        self.model = SimpleGPT2HeadTailClassifier(pretrained_model_name=model_name)
        
    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        return outputs
    
    def prepare_head_tail_inputs(self, texts, tokenizer, max_length=128):
        """
        Simplified tokenization for CI testing.
        Only processes the texts once, ignoring the tail part.
        
        Args:
            texts: List of input texts to tokenize
            tokenizer: HuggingFace tokenizer
            max_length: Maximum length for each sequence
            
        Returns:
            Dictionary of input tensors for the model
        """
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        encoding = tokenizer(
            texts, 
            max_length=max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return encoding 