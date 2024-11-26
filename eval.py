from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import math
import time

# Paths to fine-tuned model and baseline
fine_tuned_model_path = "./trained_model_duo" #trained_model_no_mask
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

# Generate responses from both models
fine_tuned_responses, fine_tuned_time = generate_responses(prompts, fine_tuned_model, tokenizer)
baseline_responses, baseline_time = generate_responses(prompts, baseline_model, tokenizer)

# Save responses for analysis
with open("evaluation_results.txt", "w") as f:
    for i, prompt in enumerate(prompts):
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Fine-Tuned Response: {fine_tuned_responses[i]}\n")
        f.write(f"Baseline Response: {baseline_responses[i]}\n")
        f.write("-" * 80 + "\n")

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

