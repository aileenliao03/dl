from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import math

# Paths to fine-tuned model and baseline
fine_tuned_model_path = "./trained_model"
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
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
val_data = dataset["validation"].select(range(10))  # Select a small subset for evaluation

# Prepare evaluation prompts
prompts = [prompt for prompt in val_data["text"] if prompt.strip()]  # Filter out empty strings

# Generate responses
def generate_responses(prompts, model, tokenizer, max_length=50):
    model.eval()  # Set model to evaluation mode
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    return responses

# Generate responses from both models
fine_tuned_responses = generate_responses(prompts, fine_tuned_model, tokenizer)
baseline_responses = generate_responses(prompts, baseline_model, tokenizer)

# Print responses
for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
    print(f"Fine-Tuned Response: {fine_tuned_responses[i]}")
    print(f"Baseline Response: {baseline_responses[i]}")
    print("-" * 80)

# Perplexity calculation
def calculate_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for text in texts:
        if not text.strip():  # Skip empty inputs
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            if not math.isfinite(loss):  # Skip unstable loss values
                print(f"Warning: Skipping unstable loss for input: {text}")
                continue
        total_loss += loss * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)
    if total_tokens == 0:  # Avoid division by zero
        return float("inf")
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

# Calculate perplexity for both models
fine_tuned_perplexity = calculate_perplexity(fine_tuned_model, tokenizer, prompts)
baseline_perplexity = calculate_perplexity(baseline_model, tokenizer, prompts)

# Print perplexity scores
print(f"Fine-Tuned Model Perplexity: {fine_tuned_perplexity}")
print(f"Baseline Model Perplexity: {baseline_perplexity}")
