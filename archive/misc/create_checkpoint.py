import torch
from transformers import AutoModelForSequenceClassification
import os

output_dir = "output/checkpoints"
os.makedirs(output_dir, exist_ok=True)

print("Loading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
state_dict = model.state_dict()

checkpoint = {
    "state_dict": state_dict,
    "config": {"model_name": "distilbert-base-uncased"}
}

print(f"Saving checkpoint to {output_dir}/distilbert_headtail_fold0.pth")
torch.save(checkpoint, f"{output_dir}/distilbert_headtail_fold0.pth")
print("Checkpoint saved successfully!") 