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
import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

# Paths to fine-tuned model and baseline
fine_tuned_model_path = "./trained_model_duo_gate_diff"  # Path to your fine-tuned model
baseline_model_path = "./trained_model_no_mask_duo_gate_diff"  # Baseline model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config first
with open(os.path.join(fine_tuned_model_path, "config.json"), "r") as f:
    config = json.load(f)
# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model_name'])
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the EOS token
model = AutoModelForCausalLM.from_pretrained(config['base_model_name'])
# Apply custom attention if this was a masked model
model = replace_attention_with_masking(model, config['use_local_mask'])
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
model.load_state_dict(state_dict)
fine_tuned_model = model.to(device)

# Load baseline model normally
baseline_model = AutoModelForCausalLM.from_pretrained(config['base_model_name'])
with open(os.path.join(baseline_model_path, "config.json"), "r") as f:
    config = json.load(f)

baseline_model = replace_attention_with_masking(baseline_model, config['use_local_mask'])
# Load the state dict
state_dict1 = torch.load(
    os.path.join(baseline_model_path, "pytorch_model.pt"),
    map_location=device,
    pickle_module=dill
)
filtered_state_dict = {
    key: value for key, value in state_dict1.items()
    if "causal_mask" not in key and "local_window_mask" not in key
}
# Load the state dict into the model
baseline_model.load_state_dict(state_dict1)
baseline_model = baseline_model.to(device)

# Set device

# Load validation dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
val_data = dataset["validation"].select(range(100))  # Use 100 samples for evaluation
prompts = [prompt for prompt in val_data["text"] if prompt.strip()]  # Filter out empty strings
max_length = 512

def generate_responses(prompts, model, tokenizer, max_length=512):
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

def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    model.eval()
    total_loss = 0
    total_tokens = 0
x = torch.randn(1000, 1000, device='cuda')

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    fine_tuned_model.eval()
    baseline_model.eval()

    prompt = prompts[1]  # Use the second prompt as an example

    # Fine-tuned model inference
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with record_function("Fine-Tuned Model Inference"):
        outputs_fine_tuned = fine_tuned_model.generate(**inputs, max_new_tokens=max_length)

    # Baseline model inference
    with record_function("Baseline Model Inference"):
        outputs_baseline = baseline_model.generate(**inputs, max_new_tokens=max_length)

# Export profiler data to a file
prof.export_chrome_trace("./trace.json")

print("Profiler trace exported to './trace.json'")

