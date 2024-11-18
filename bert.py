from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Load BERT model and tokenizer
model_name = "bert-base-uncased"  # Use a general BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda")

# Input text with a mask token
input_text = "Apple Banana and [MASK]."

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted token for [MASK]
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
predicted_token_id = logits[0, mask_token_index, :].argmax(axis=-1).item()
predicted_word = tokenizer.decode(predicted_token_id)

print(f"Next word: {predicted_word}")
