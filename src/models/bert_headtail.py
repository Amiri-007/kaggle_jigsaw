import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel, BertConfig, AutoModel
from typing import Optional, Tuple, Dict, Any, List, Union

class BertHeadTailClassifier(BertPreTrainedModel):
    """BERT model for classification using both head and tail of the input sequence.
    
    This model concatenates the first 128 tokens and last 128 tokens of the input text
    to handle longer sequences more efficiently.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = 1  # Binary classification for toxicity
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Double the classifier input size to accommodate both head and tail embeddings
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        head_input_ids: Optional[torch.Tensor] = None,
        head_attention_mask: Optional[torch.Tensor] = None,
        head_token_type_ids: Optional[torch.Tensor] = None,
        tail_input_ids: Optional[torch.Tensor] = None,
        tail_attention_mask: Optional[torch.Tensor] = None,
        tail_token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_position_ids: Optional[torch.Tensor] = None,
        tail_position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Process head portion through BERT
        head_outputs = self.bert(
            input_ids=head_input_ids,
            attention_mask=head_attention_mask,
            token_type_ids=head_token_type_ids,
            position_ids=head_position_ids,
            return_dict=True,
        )
        
        # Process tail portion through BERT
        tail_outputs = self.bert(
            input_ids=tail_input_ids,
            attention_mask=tail_attention_mask,
            token_type_ids=tail_token_type_ids,
            position_ids=tail_position_ids,
            return_dict=True,
        )
        
        # Get the [CLS] token embedding for both head and tail
        head_pooled_output = head_outputs.pooler_output
        tail_pooled_output = tail_outputs.pooler_output
        
        # Concatenate head and tail representations
        pooled_output = torch.cat([head_pooled_output, tail_pooled_output], dim=1)
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
            "hidden_states": head_outputs.hidden_states,
            "attentions": head_outputs.attentions,
        }


class BertHeadTailForSequenceClassification(nn.Module):
    """Wrapper class for BertHeadTailClassifier to simplify usage."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 1):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.model = BertHeadTailClassifier.from_pretrained(model_name, config=self.config)
        
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
        full_encodings = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        
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
        tail_token_type_ids = []
        
        for i, text in enumerate(texts):
            # Get full token IDs for this text
            full_ids = full_encodings["input_ids"][i]
            
            if len(full_ids) <= max_length:
                # If text is short, use the same tokens but with different positional encoding
                tail_ids = full_ids
            else:
                # Extract last max_length-2 tokens (account for [CLS] and [SEP])
                tail_ids = full_ids[-(max_length-2):]
            
            # Add special tokens
            tail_ids = [tokenizer.cls_token_id] + list(tail_ids) + [tokenizer.sep_token_id]
            
            # Pad to max_length
            padding_length = max_length - len(tail_ids)
            tail_ids = tail_ids + [tokenizer.pad_token_id] * padding_length
            
            # Create attention mask
            tail_mask = [1] * (max_length - padding_length) + [0] * padding_length
            
            # Create token type IDs (all 0 for single segment)
            tail_token_type = [0] * max_length
            
            tail_input_ids.append(tail_ids)
            tail_attention_masks.append(tail_mask)
            tail_token_type_ids.append(tail_token_type)
        
        # Convert to tensors
        model_inputs = {
            "head_input_ids": head_inputs["input_ids"],
            "head_attention_mask": head_inputs["attention_mask"],
            "head_token_type_ids": head_inputs["token_type_ids"],
            "tail_input_ids": torch.tensor(tail_input_ids),
            "tail_attention_mask": torch.tensor(tail_attention_masks),
            "tail_token_type_ids": torch.tensor(tail_token_type_ids),
        }
        
        return model_inputs 

class SimpleBertHeadTailClassifier(nn.Module):
    """
    Simplified BERT classifier for CI testing.
    Uses a small pre-trained model and freezes weights.
    """
    def __init__(self, pretrained_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # Freeze BERT parameters for faster training in CI
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Simple classifier head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_input_ids: Optional[torch.Tensor] = None,
        head_attention_mask: Optional[torch.Tensor] = None,
        head_token_type_ids: Optional[torch.Tensor] = None,
        tail_input_ids: Optional[torch.Tensor] = None,
        tail_attention_mask: Optional[torch.Tensor] = None,
        tail_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Use either the standard input_ids or head_input_ids based on what's provided
        if input_ids is not None:
            ids = input_ids
            mask = attention_mask
            token_types = token_type_ids
        else:
            ids = head_input_ids
            mask = head_attention_mask
            token_types = head_token_type_ids
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_types,
            return_dict=True
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Apply sigmoid for probability output
        probs = torch.sigmoid(logits)
        
        return {
            "logits": logits.squeeze(-1),
            "probs": probs.squeeze(-1)
        }

class SimpleBertHeadTailForSequenceClassification(nn.Module):
    """Wrapper class for SimpleBertHeadTailClassifier to simplify usage."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 1):
        super().__init__()
        self.model = SimpleBertHeadTailClassifier(pretrained_model_name=model_name)
        
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
        encoding = tokenizer(
            texts, 
            max_length=max_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return encoding 