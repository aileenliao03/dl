from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import math
import time

# Paths to fine-tuned model and baseline
fine_tuned_model_path = "./trained_model_with_learned_mask" #trained_model_no_mask
baseline_model_path = "meta-llama/Llama-3.2-1B"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model.to(device)
baseline_model.to(device)

# Load validation dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
val_data = dataset["validation"].select(range(100))  # Use 100 samples for evaluation
prompts = [prompt for prompt in val_data["text"] if prompt.strip()]  # Filter out empty strings


# Generate responses
def generate_responses(prompts, model, tokenizer, max_length=50):
    model.eval()
    responses = []
    start_time = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    total_time = time.time() - start_time
    avg_time_per_response = total_time / len(prompts)
    return responses, avg_time_per_response


# Perplexity calculation
def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    perplexity = math.exp(total_loss / total_tokens)
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


# Create long-context prompts by concatenating entries from WikiText
long_context_prompts = []
current_context = ""
max_context_length = 8192  # Target long input length (in tokens)

for text in val_data["text"]:
    if len(current_context.split()) + len(text.split()) <= max_context_length:
        current_context += " " + text
    else:
        long_context_prompts.append(current_context.strip())
        current_context = text

# Add the final accumulated context
if current_context:
    long_context_prompts.append(current_context.strip())

# Limit to a manageable subset for evaluation
long_context_prompts = long_context_prompts[:10]  # Adjust as needed

# Evaluate fine-tuned and baseline models
long_fine_tuned_responses, long_fine_tuned_time = generate_responses(long_context_prompts, fine_tuned_model, tokenizer, max_length=1024)
long_baseline_responses, long_baseline_time = generate_responses(long_context_prompts, baseline_model, tokenizer, max_length=1024)

# Calculate BLEU and ROUGE
long_fine_tuned_bleu = calculate_bleu(long_context_prompts, long_fine_tuned_responses)
long_baseline_bleu = calculate_bleu(long_context_prompts, long_baseline_responses)
long_fine_tuned_rouge = calculate_rouge(long_context_prompts, long_fine_tuned_responses)
long_baseline_rouge = calculate_rouge(long_context_prompts, long_baseline_responses)

# Print results
print(f"Fine-Tuned Model Long BLEU: {long_fine_tuned_bleu}")
print(f"Baseline Model Long BLEU: {long_baseline_bleu}")
print(f"Fine-Tuned Model Long ROUGE: {long_fine_tuned_rouge}")
print(f"Baseline Model Long ROUGE: {long_baseline_rouge}")
print(f"Fine-Tuned Long Avg Inference Time: {long_fine_tuned_time:.4f} seconds per response")
print(f"Baseline Long Avg Inference Time: {long_baseline_time:.4f} seconds per response")

import torch.profiler as profiler

# Profile inference for fine-tuned model
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=profiler.tensorboard_trace_handler("./fine_tuned_profile"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for prompt in prompts[:10]:  # Profile a subset for efficiency
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            fine_tuned_model.generate(**inputs, max_new_tokens=50)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Profile inference for baseline model
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=profiler.tensorboard_trace_handler("./baseline_profile"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for prompt in prompts[:10]:  # Profile a subset for efficiency
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            baseline_model.generate(**inputs, max_new_tokens=50)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
