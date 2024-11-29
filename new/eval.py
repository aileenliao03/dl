from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from torch.utils.data import DataLoader
from datasets import load_dataset,  DatasetDict, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import wandb
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import time
login(token="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")
wandb.login(key="a1df0c88c10ee7966670142e4afd5a895138c7cf")
device =  ('cuda' if torch.cuda.is_available() else 'cpu')

EXAMPLES = 100
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
val_data = dataset["validation"].select(range(EXAMPLES))
prompts = [prompt for prompt in val_data["text"] if prompt.strip()]  # Filter out empty strings
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
#baseline_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
# Generate responses
def load_model(name,device ):
  path = "/content/drive/My Drive/llama/models/" + name+"/"
  forget_layer = torch.load(path + "forget.pt")
  base_llm_model = AutoModelForCausalLM.from_pretrained(path).to(device)
  return base_llm_model, forget_layer

import time
def generate_responses(prompts, model, tokenizer, max_length=50, custom= False):
    model.eval()
    responses = []
    start_time = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            if custom:
              outputs = model.generate(**inputs, max_length=max_length)
            else:
              outputs = model.generate(**inputs, max_new_tokens=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    total_time = time.time() - start_time
    avg_time_per_response = total_time / len(prompts)
    return responses, avg_time_per_response
# Perplexity calculation
def calculate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512, custom = False):
    model.eval()
    total_loss = 0
    total_tokens = 0
    notactive_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            if custom:
              loss= outputs["loss"].item()
              if "count" in outputs:
                notactive_tokens += outputs["count"]
            else:
              loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    ratio = (total_tokens -notactive_tokens)/total_tokens
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity, ratio
name = "F_GRU_TFTFF2step_10000_Schedule_Reinforce_Step_MLPForget_Full_Fintune"
base_llm_model, forget_layer = load_model(name, device)
#choice_config = None
choice_config= {"percent":.90}
#choice_config= {"threshold":.5}
fine_tuned_model = Language_Model(base_llm_model, forget_layer, soft=False, choice_config=choice_config).to(device).eval()
fine_tuned_responses, fine_tuned_time = generate_responses(prompts, fine_tuned_model, tokenizer, custom= True)

fine_tuned_perplexity, fine_tuned_ratio = calculate_perplexity(fine_tuned_model, tokenizer, prompts, custom=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_bleu(references, hypotheses):
    """
    Calculate BLEU score for a set of references and hypotheses.

    Args:
        references (list of str): Ground truth text.
        hypotheses (list of str): Generated text by the model.

    Returns:
        float: Average BLEU score.
    """
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_function)
        bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def calculate_rouge(references, hypotheses):
    """
    Calculate ROUGE scores for a set of references and hypotheses.

    Args:
        references (list of str): Ground truth text.
        hypotheses (list of str): Generated text by the model.

    Returns:
        dict: Average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

    return {key: sum(values) / len(values) if values else 0.0 for key, values in rouge_scores.items()}
# Calculate BLEU and ROUGE
fine_tuned_bleu = calculate_bleu(prompts, fine_tuned_responses)
#baseline_bleu = calculate_bleu(prompts, baseline_responses)
fine_tuned_rouge = calculate_rouge(prompts, fine_tuned_responses)
#baseline_rouge = calculate_rouge(prompts, baseline_responses)

print(f"Fine-Tuned Model Perplexity: {fine_tuned_perplexity}")
#print(f"Fine-Tuned Model Ratio: {fine_tuned_ratio}")
#print(f"Baseline Model Perplexity: {baseline_perplexity}")
#print(f"Fine-Tuned Avg Inference Time: {fine_tuned_time:.4f} seconds per response")
#print(f"Baseline Avg Inference Time: {baseline_time:.4f} seconds per response")
print(f"Fine-Tuned Model BLEU: {fine_tuned_bleu}")
#print(f"Baseline Model BLEU: {baseline_bleu}")
print(f"Fine-Tuned Model ROUGE: {fine_tuned_rouge}")
#print(f"Baseline Model ROUGE: {baseline_rouge}")

# Log results to WandB
wandb.init(
    project="llama_eval",  # Replace with your WandB project name
    entity="ai0liao",
    config={
        "description": "Evaluation of fine-tuned vs baseline language models",
    }
)
logs = {
    "name": name+"90percent",
    "Fine-Tuned Model Perplexity": fine_tuned_perplexity,
   # "Baseline Model Perplexity": baseline_perplexity,
    "Fine-Tuned Avg Inference Time (s)": fine_tuned_time,
   # "Baseline Avg Inference Time (s)": baseline_time,
    "Fine-Tuned Model BLEU": fine_tuned_bleu,
   # "Baseline Model BLEU": baseline_bleu,
    "Fine-Tuned Model ROUGE": fine_tuned_rouge,
   # "Baseline Model ROUGE": baseline_rouge
}
if choice_config != None:
  logs.update(choice_config)
#wandb.log(logs)