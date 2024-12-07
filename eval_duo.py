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
import torch.profiler
from torch.utils.data import DataLoader
from datasets import Dataset

# Paths to fine-tuned model and baseline
baseline_model_path = "./trained_model_no_mask_duo_gate_diff"  # Path to your fine-tuned model
fine_tuned_model_path = "./trained_model_duo_gate_diff"  # Baseline model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config first
with open(os.path.join(fine_tuned_model_path, "config.json"), "r") as f:
    config = json.load(f)
# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the EOS token
fine_tuned_model = AutoModelForCausalLM.from_pretrained(config['base_model_name'])
# Apply custom attention if this was a masked model
fine_tuned_model = replace_attention_with_masking(fine_tuned_model, config['use_local_mask'])
# Load the state dict
state_dict = torch.load(
    os.path.join(fine_tuned_model_path, "pytorch_model.pt"),
    map_location=device,
    pickle_module=dill
)

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


def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    data_loader = DataLoader(texts, batch_size=batch_size, shuffle=False)

    for batch_texts in data_loader:
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss =  outputs.loss.item()
            
            # Count valid tokens (excluding padding tokens)
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return perplexity

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

def evaluate(model_name, model, prompts, tokenizer, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0
    gate_means = []
    gate_stds = []
    sparsities = []
    responses = []
    total_batches = 0
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

        # print(f"Batch {total_batches + 1}:")
        # for prompt in batch_prompts:
        #     print(f"Prompt: {prompt}")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length)
            loss = 0 #outputs.loss.item()
        batch_time = time.time() - start_time
        total_time += batch_time
        total_batches +=1
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)

        # Collect statistics for gating and sparsity
        for name, module in model.named_modules():
            if isinstance(module, GatedDuoAttention):
                gate_means.append(module.last_gate_mean)
                gate_stds.append(module.last_gate_std)
                sparsities.append(module.sparsity)
    
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_gate_mean = np.mean(gate_means) if gate_means else 0
    avg_gate_std = np.mean(gate_stds) if gate_stds else 0
    avg_sparsity = np.mean(sparsities) if sparsities else 0
    perplexity = calculate_perplexity(model, tokenizer, prompts, batch_size=batch_size, max_length=max_length)
    bleu_score = calculate_bleu(prompts, responses)
    rouge_scores = calculate_rouge(prompts, responses)

    plt.figure(figsize=(10, 6))
    # Plot gate means
    plt.subplot(2, 1, 1)
    plt.plot(gate_means, label='Gate Means')
    plt.xlabel('Batch')
    plt.ylabel('Mean Gate Value')
    plt.title('Gate Means Across Batches')
    plt.legend()

    # Plot gate standard deviations
    plt.subplot(2, 1, 2)
    plt.plot(gate_stds, label='Gate Standard Deviations', color='orange')
    plt.xlabel('Batch')
    plt.ylabel('Gate Std Value')
    plt.title('Gate Standard Deviations Across Batches')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_eval.png")
    
    # Print results for this model
    print(f"\nResults for {model_name} Model:")
    print(f"Average Loss: {avg_loss}")
    print(f"Average Gate Mean: {avg_gate_mean}")
    print(f"Average Gate Std: {avg_gate_std}")
    print(f"Average Sparsity: {avg_sparsity}")
    print(f"Perplexity: {perplexity}")
    print(f"BLEU Score: {bleu_score}")
    print(f"ROUGE Scores: {rouge_scores}")

    return avg_loss, avg_gate_mean, avg_gate_std, avg_sparsity, perplexity, bleu_score, rouge_scores, sparsities

# Evaluate both models
print("Evaluating Fine-Tuned Model...")
avg_loss_ft, avg_gate_mean_ft, avg_gate_std_ft, avg_sparsity_ft, perplexity, bleu_score, rouge_scores, sparsities_ft = evaluate("fine_tuned", fine_tuned_model, prompts, tokenizer)

print("Evaluating Baseline Model...")
avg_loss_baseline, avg_gate_mean_baseline, avg_gate_std_baseline, avg_sparsity_baseline, perplexity, bleu_score, rouge_scores, sparsities_baseline = evaluate("baseline", baseline_model, prompts, tokenizer)


plt.clf()
plt.plot(sparsities_ft, label='Fine Tuned', color='green')
plt.plot(sparsities_baseline, label='Baseline', color='red')
plt.xlabel('Batch')
plt.ylabel('Sparsity Value')
plt.title('Sparsity Across Batches')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("sparsities_eval.png")