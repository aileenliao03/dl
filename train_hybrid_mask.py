from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
import os

# Configuration
model_name = "meta-llama/Llama-3.2-1B"
use_hybrid_attention = True
learning_rate = 3e-5
num_epochs = 4
batch_size = 12
accumulation_steps = 4
max_seq_len = 512
window_size = 128

# Tokenizer and Model Initialization
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

class HierarchicalMaskedAttention(nn.Module):
    def __init__(self, base_attention, levels=3, downsampling_factor=2):
        super().__init__()
        self.base_attention = base_attention
        self.levels = levels
        self.downsampling_factor = downsampling_factor

    def downsample(self, memory):
        batch_size, seq_len, hidden_dim = memory.size()
        seq_len_new = seq_len // self.downsampling_factor
        memory = memory[:, :seq_len_new * self.downsampling_factor, :]
        memory = memory.view(batch_size, seq_len_new, self.downsampling_factor, hidden_dim).mean(dim=2)
        return memory

    def forward(self, query, key, value, attention_mask=None):
        all_keys, all_values = [key], [value]
        for _ in range(self.levels - 1):
            key = self.downsample(key)
            value = self.downsample(value)
            all_keys.append(key)
            all_values.append(value)

        outputs = []
        for k, v in zip(all_keys, all_values):
            outputs.append(self.base_attention(query, k, v, attn_mask=attention_mask)[0])

        return sum(outputs) / len(outputs)

class SlidingWindowAttention(nn.Module):
    def __init__(self, base_attention, window_size=128, global_indices=[]):
        super().__init__()
        self.base_attention = base_attention
        self.window_size = window_size
        self.global_indices = global_indices

    def forward(self, query, key, value, attention_mask=None):
        seq_len = query.size(1)
        local_attention_outputs = []

        for i in range(0, seq_len, self.window_size):
            start, end = i, min(i + self.window_size, seq_len)
            local_query, local_key, local_value = query[:, start:end], key[:, start:end], value[:, start:end]
            local_attention_outputs.append(self.base_attention(local_query, local_key, local_value)[0])

        combined_output = torch.cat(local_attention_outputs, dim=1)

        if self.global_indices:
            global_query = query[:, self.global_indices]
            global_key = key[:, self.global_indices]
            global_value = value[:, self.global_indices]
            global_attention_output = self.base_attention(global_query, global_key, global_value)[0]
            combined_output[:, self.global_indices] = global_attention_output

        return combined_output

class HybridMaskedAttention(nn.Module):
    def __init__(self, base_attention, levels=3, downsampling_factor=2, max_seq_len=512, window_size=128):
        super().__init__()
        self.hierarchical_attention = HierarchicalMaskedAttention(base_attention, levels, downsampling_factor)
        self.soft_mask = nn.Parameter(torch.ones(max_seq_len))
        self.sigmoid = nn.Sigmoid()
        self.sliding_window_attention = SlidingWindowAttention(base_attention, window_size)

    def forward(self, query, key, value, attention_mask=None):
        hierarchical_output = self.hierarchical_attention(query, key, value, attention_mask)
        soft_mask = self.sigmoid(self.soft_mask[:query.size(1)])
        sliding_window_output = self.sliding_window_attention(query, key, value, attention_mask)
        return (hierarchical_output + sliding_window_output) * soft_mask.unsqueeze(0).unsqueeze(-1)

def replace_attention_with_masking(model, levels=3, downsampling_factor=2):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(f"Replacing attention module: {name}")
            hierarchical_attention = HybridMaskedAttention(module, levels, downsampling_factor)
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, module_name, hierarchical_attention)
    return model

if use_hybrid_attention:
    model = replace_attention_with_masking(model)


dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_data = dataset["train"].select(range(10000))
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_seq_len
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
train_loader = DataLoader(tokenized_dataset, batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

val_data = dataset["validation"]
val_tokenized = val_data.map(tokenize_function, batched=True, remove_columns=["text"])
val_tokenized.set_format(type="torch", columns=["input_ids"])
val_loader = DataLoader(val_tokenized, batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)
num_training_steps = len(train_loader) * 4
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

scaler = GradScaler()
model.train()

print(f"Number of samples in train_data: {len(train_data)}")  # Should print: 1


for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        inputs = {key: val.to(device) for key, val in batch.items()}

        with autocast():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}")


print("Training completed.")

output_dir = "./trained_hybrid_model"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

