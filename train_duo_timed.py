from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
import time
from contextlib import contextmanager
import torch.nn.functional as F

mask = True
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

class DuoAttention(nn.Module):
    def __init__(self, base_attention, window_size=128):
        super().__init__()
        self.base_attention = base_attention
        self.window_size = window_size
        
        # Get hidden dim from base attention
        hidden_dim = base_attention.embed_dim
        
        # Simple MLP with one layer and leaky ReLU
        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, query, key, value, attention_mask=None):
        # Get mask choice from last query vector
        last_query = query[:, -1]  # Shape: [batch_size, hidden_dim]
        mask_choice = self.mask_mlp(last_query)  # Shape: [batch_size, 1]
        
        # Original attention
        full_output = self.base_attention(query, key, value, attn_mask=attention_mask)[0]
        
        # Localized attention on recent queries
        seq_len = query.size(1)
        start = max(0, seq_len - self.window_size)
        local_query = query[:, start:seq_len]
        local_key = key[:, start:seq_len] 
        local_value = value[:, start:seq_len]
        local_output = self.base_attention(local_query, local_key, local_value, attn_mask=attention_mask)[0]
        
        # Interpolate between full and local attention based on mask choice
        mask_choice = mask_choice.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
        output = torch.where(mask_choice < 0.1, 
                           (1 - mask_choice) * local_output,
                           mask_choice * full_output + (1 - mask_choice) * local_output)
        return output

def replace_attention_with_masking(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(f"Replacing attention module: {name}")
            hierarchical_attention = DuoAttention(module)
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, module_name, hierarchical_attention)
    return model

if mask:
    model = replace_attention_with_masking(model)

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_data = dataset["train"].select(range(10000))
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


tokenized_dataset = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids"])
train_loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

val_data = dataset["validation"]
val_tokenized = val_data.map(tokenize_function, batched=True, remove_columns=["text"])
val_tokenized.set_format(type="torch", columns=["input_ids"])
val_loader = DataLoader(val_tokenized, batch_size=4)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)
num_training_steps = len(train_loader) * 4
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

#scaler = GradScaler()
accumulation_steps = 4
model.train()

@contextmanager
def timer():
    start = time.perf_counter()
    try:
        yield start
    finally:
        end = time.perf_counter()
        elapsed_time = end - start

class TimedForwardModel(nn.Module):
    def __init__(self, model, inference_weight=0.01, target_time=0.1):
        super().__init__()
        self.model = model
        self.inference_weight = inference_weight
        self.target_time = target_time
    
    def forward(self, **inputs):
        # Time the forward pass
        start_time = time.perf_counter()
        outputs = self.model(**inputs)
        forward_time = time.perf_counter() - start_time
        
        # Add timing penalty to loss
        if isinstance(outputs, tuple):
            original_loss = outputs[0]
        else:
            original_loss = outputs.loss
            
        # Compute time penalty (using smooth L1 loss for stability)
        time_tensor = torch.tensor(forward_time, device=original_loss.device, dtype=torch.float32)
        target_tensor = torch.tensor(self.target_time, device=original_loss.device, dtype=torch.float32)
        time_penalty = F.smooth_l1_loss(time_tensor, target_tensor)
        
        # Combine losses
        total_loss = original_loss + self.inference_weight * time_penalty
        
        if hasattr(outputs, '_replace'):  # For named tuples
            outputs = outputs._replace(loss=total_loss)
        else:  # For other objects
            outputs.loss = total_loss
            
        return outputs

# Wrap your model with timing
model = TimedForwardModel(model, inference_weight=0.1, target_time=0.001)
model.to(device)

for epoch in range(4):
    for i, batch in enumerate(train_loader):
        try:
            inputs = {key: val.to(device) for key, val in batch.items()}
            
            with torch.amp.autocast('cuda'):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / accumulation_steps
                
                # Get original loss for logging
                original_loss = outputs.loss - loss.item()
                time_loss = loss.item() - original_loss

            #scaler.scale(loss).backward()
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()  # Move optimizer step before scheduler step
                lr_scheduler.step()
                #scaler.update()
                optimizer.zero_grad()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Total Loss: {loss.item():.4f}, "
                      f"Original Loss: {original_loss:.4f}, Time Loss: {time_loss:.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e) or "invalid argument" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                print(f"CUDA error encountered. Skipping batch...")
                continue
            else:
                raise e

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}")


print("Training completed.")

if mask:
    output_dir = "./trained_model_duo_timed"
else:
    output_dir_no_mask = "./trained_model_no_mask_duo_timed"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

