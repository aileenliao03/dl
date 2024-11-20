from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler

mask = True
learning_rate = 3e-5
num_epochs = 4
batch_size = 12
accumulation_steps = 4
regularization_weight = 0.01

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

class HierarchicalMaskedAttention(nn.Module):
    def __init__(self, base_attention, levels=3, downsampling_factor=2):
        super().__init__()
        self.base_attention = base_attention
        self.levels = levels
        self.downsampling_factor = downsampling_factor
        self.learned_weights = nn.Parameter(torch.ones(levels))

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
        for level, (k, v) in enumerate(zip(all_keys, all_values)):
            attention_output = self.base_attention(query, k, v, attn_mask=attention_mask)[0]
            outputs.append(attention_output * self.learned_weights[level])

        total_weight = self.learned_weights.sum()
        return sum(outputs) / total_weight

def replace_attention_with_learned_masking(model, levels=3, downsampling_factor=2):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(f"Replacing attention module: {name}")
            hierarchical_attention = HierarchicalMaskedAttention(module, levels, downsampling_factor)
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, module_name, hierarchical_attention)
    return model

if mask:
    model = replace_attention_with_learned_masking(model)

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_data = dataset["train"].select(range(10000))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids"])
train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

val_data = dataset["validation"]
val_tokenized = val_data.map(tokenize_function, batched=True, remove_columns=["text"])
val_tokenized.set_format(type="torch", columns=["input_ids"])
val_loader = DataLoader(val_tokenized, batch_size=batch_size)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def calculate_regularization_loss(model):
    reg_loss = 0.0
    for module in model.modules():
        if isinstance(module, HierarchicalMaskedAttention):
            reg_loss += torch.norm(module.learned_weights, p=1)
    return reg_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

scaler = GradScaler()

model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        inputs = {key: val.to(device) for key, val in batch.items()}

        with autocast():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / accumulation_steps

            reg_loss = calculate_regularization_loss(model)
            loss += regularization_weight * reg_loss

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

    for name, module in model.named_modules():
        if isinstance(module, HierarchicalMaskedAttention):
            print(f"Learned Weights for {name}: {module.learned_weights.data}")

output_dir = "./trained_model_with_learned_mask" if mask else "./trained_model_no_mask"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
