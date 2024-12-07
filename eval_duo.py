from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import math
import time
import os
import json
import dill
from train_duo_gate_diff import (
    GatedDuoAttention,
    create_causal_mask,
    create_local_window_mask,
    replace_attention_with_masking
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from datasets import Dataset

# Paths to fine-tuned model and baseline
baseline_model_path = "./trained_model_duo_gate_diff"  # Path to your fine-tuned model
fine_tuned_model_path = "./trained_model_no_mask_duo_gate_diff"  # Baseline model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config first
with open(os.path.join(fine_tuned_model_path, "config.json"), "r") as f:
    config = json.load(f)
# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'], padding_side='left')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the EOS token
tokenizer.padding_side='left'
fine_tuned_model = AutoModelForCausalLM.from_pretrained(config['base_model_name'])
# Apply custom attention if this was a masked model
fine_tuned_model = replace_attention_with_masking(fine_tuned_model, config['use_local_mask'])
# Load the state dict
state_dict = torch.load(
    os.path.join(fine_tuned_model_path, "pytorch_model.pt"),
    map_location=device,
    pickle_module=dill
)
#print(state_dict.items())
#filtered_state_dict = {
#    key: value for key, value in state_dict.items()
#    if "causal_mask" not in key and "local_window_mask" not in key
#}

# Load the state dict into the model
fine_tuned_model.load_state_dict(state_dict)
fine_tuned_model = fine_tuned_model.to(device)

# Load baseline model normally
baseline_model = AutoModelForCausalLM.from_pretrained(config['base_model_name'])
with open(os.path.join(baseline_model_path, "config.json"), "r") as f:
    config = json.load(f)

baseline_model = replace_attention_with_masking(baseline_model, True)
# Load the state dict
state_dict1 = torch.load(
    os.path.join(baseline_model_path, "pytorch_model.pt"),
    map_location=device,
    pickle_module=dill
)

# Load the state dict into the model
baseline_model.load_state_dict(state_dict1)
baseline_model = baseline_model.to(device)

# Load validation dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
val_data = dataset["validation"].select(range(100))  # Use 100 samples for evaluation
prompts = [prompt for prompt in val_data["text"] if prompt.strip()]  # Filter out empty strings

def generate_responses(prompts, model, tokenizer, max_length=512, batch_size=8):
    model.eval()
    responses = []
    total_time = 0
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:min(i + batch_size, len(prompts))]
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=max_length
        ).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length)
        batch_time = time.time() - start_time
        total_time += batch_time
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)
    
    avg_time_per_response = total_time / len(prompts)
    return responses, avg_time_per_response

def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Make sure model knows about pad token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

def evaluate(model, dataset, tokenizer, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0
    gate_means = []
    gate_stds = []
    sparsities = []

    # Split the dataset into batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate number of batches

    with torch.no_grad():
        for i in range(num_batches):
            # Extract the batch of texts from the dataset
            batch_data = dataset[i * batch_size: (i + 1) * batch_size]
            batch_texts = batch_data["text"]  # Extract the "text" field for the batch

            # Tokenize the batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            ).to(device)

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss

            # Collect the gate statistics and sparsity
            for name, module in model.named_modules():
                if isinstance(module, GatedDuoAttention):
                    gate_means.append(module.last_gate_mean)
                    gate_stds.append(module.last_gate_std)
                    sparsities.append(module.sparsity)

    avg_loss = total_loss / num_batches
    avg_gate_mean = np.mean(gate_means) if gate_means else 0
    avg_gate_std = np.mean(gate_stds) if gate_stds else 0
    avg_sparsity = np.mean(sparsities) if sparsities else 0

    # Plot the means, stds, and sparsities
    plt.figure(figsize=(10, 6))
    
    # Plot gate means
    plt.subplot(3, 1, 1)
    plt.plot(gate_means, label='Gate Means')
    plt.xlabel('Batch')
    plt.ylabel('Mean Gate Value')
    plt.title('Gate Means Across Batches')
    plt.legend()
    
    # Plot gate standard deviations
    plt.subplot(3, 1, 2)
    plt.plot(gate_stds, label='Gate Standard Deviations', color='orange')
    plt.xlabel('Batch')
    plt.ylabel('Gate Std Value')
    plt.title('Gate Standard Deviations Across Batches')
    plt.legend()

    # Plot sparsity
    plt.subplot(3, 1, 3)
    plt.plot(sparsities, label='Sparsity', color='green')
    plt.xlabel('Batch')
    plt.ylabel('Sparsity Value')
    plt.title('Sparsity Across Batches')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return avg_loss, avg_gate_mean, avg_gate_std, avg_sparsity


# Now, you can run the evaluation and plot the results
avg_loss, avg_gate_mean, avg_gate_std, avg_sparsity = evaluate(fine_tuned_model, val_data, tokenizer)

# Optionally, print out the average values
print(f"Average Loss: {avg_loss}")
print(f"Average Gate Mean: {avg_gate_mean}")
print(f"Average Gate Std: {avg_gate_std}")
print(f"Average Sparsity: {avg_sparsity}")