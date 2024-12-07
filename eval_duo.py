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

# Paths to fine-tuned model and baseline
fine_tuned_model_path = "./trained_model_duo_gate_diff_1204"  # Path to your fine-tuned model
baseline_model_path = "./trained_model_no_mask_duo_gate_diff"  # Baseline model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config first
with open(os.path.join(fine_tuned_model_path, "config.json"), "r") as f:
    config = json.load(f)
# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])
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

# Set device

# Load validation dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
val_data = dataset["validation"].select(range(100))  # Use 100 samples for evaluation
prompts = [prompt for prompt in val_data["text"] if prompt.strip()]  # Filter out empty strings
max_length = 512

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


# Generate responses from both models
fine_tuned_responses, fine_tuned_time = generate_responses(prompts, fine_tuned_model, tokenizer, batch_size=8)
baseline_responses, baseline_time = generate_responses(prompts, baseline_model, tokenizer, batch_size=8)

# Save responses for analysis
with open("evaluation_results.txt", "w") as f:
    for i, prompt in enumerate(prompts):
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Fine-Tuned Response: {fine_tuned_responses[i]}\n")
        f.write(f"Baseline Response: {baseline_responses[i]}\n")
        f.write("-" * 80 + "\n")

# Calculate perplexity for both models
fine_tuned_perplexity = calculate_perplexity(fine_tuned_model, tokenizer, prompts)
baseline_perplexity = calculate_perplexity(baseline_model, tokenizer, prompts)

# BLEU Score Calculation
def calculate_bleu(references, candidates):
    scores = [sentence_bleu([ref.split()], cand.split()) for ref, cand in zip(references, candidates)]
    return sum(scores) / len(scores)

# ROUGE Score Calculation
def calculate_rouge(references, candidates):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
    avg_scores = {
        "rouge1": sum(score["rouge1"].fmeasure for score in scores) / len(scores),
        "rouge2": sum(score["rouge2"].fmeasure for score in scores) / len(scores),
        "rougeL": sum(score["rougeL"].fmeasure for score in scores) / len(scores),
    }
    return avg_scores

# Calculate BLEU and ROUGE
fine_tuned_bleu = calculate_bleu(prompts, fine_tuned_responses)
baseline_bleu = calculate_bleu(prompts, baseline_responses)
fine_tuned_rouge = calculate_rouge(prompts, fine_tuned_responses)
baseline_rouge = calculate_rouge(prompts, baseline_responses)

# Print Results
print(f"Fine-Tuned Model Perplexity: {fine_tuned_perplexity}")
print(f"Baseline Model Perplexity: {baseline_perplexity}")
print(f"Fine-Tuned Model BLEU: {fine_tuned_bleu}")
print(f"Baseline Model BLEU: {baseline_bleu}")
print(f"Fine-Tuned Model ROUGE: {fine_tuned_rouge}")
print(f"Baseline Model ROUGE: {baseline_rouge}")
print(f"Fine-Tuned Avg Inference Time: {fine_tuned_time:.4f} seconds per response")
print(f"Baseline Avg Inference Time: {baseline_time:.4f} seconds per response")
