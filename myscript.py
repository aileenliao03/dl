from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

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

def replace_attention_with_masking(model, levels=3, downsampling_factor=2):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            hierarchical_attention = HierarchicalMaskedAttention(module, levels, downsampling_factor)
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, module_name, hierarchical_attention)
    return model

model = replace_attention_with_masking(model)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"]
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

tokenized_dataset = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids"])
train_loader = DataLoader(tokenized_dataset, batch_size=6, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)
num_training_steps = len(train_loader) * 4
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

scaler = GradScaler()
accumulation_steps = 4
model.train()

for epoch in range(4):
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        inputs = {key: val.to(device) for key, val in batch.items()}
        
        with autocast():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

print("Training completed.")

output_dir = "./trained_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

