from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)
from typing import Optional, Tuple
import math

mask = True
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

class DuoAttention(nn.Module):
    def __init__(self, base_attention, window_size=128):
        super().__init__()
        self.base_attention = base_attention
        self.window_size = window_size
        
        # Get attributes from base attention
        self.num_heads = base_attention.num_heads
        self.num_key_value_heads = base_attention.num_key_value_heads
        self.head_dim = base_attention.head_dim
        self.hidden_size = base_attention.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Copy projections from base attention
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.rotary_emb = base_attention.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz_x_2, q_len, _ = hidden_states.size()
        assert bsz_x_2 % 2 == 0
        bsz = bsz_x_2 // 2

        # Split into full and local attention inputs
        full_hidden_states = hidden_states[:bsz]
        local_hidden_states = hidden_states[bsz:]

        # Project states for full attention
        with torch.no_grad():
            full_query_states = self.q_proj(full_hidden_states)
            full_key_states = self.k_proj(full_hidden_states)
            full_value_states = self.v_proj(full_hidden_states)
            
            full_query_states = full_query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            full_key_states = full_key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            full_value_states = full_value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Project states for local attention
        local_query_states = self.q_proj(local_hidden_states)
        local_key_states = self.k_proj(local_hidden_states)
        local_value_states = self.v_proj(local_hidden_states)
        
        local_query_states = local_query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        local_key_states = local_key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        local_value_states = local_value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(full_value_states, position_ids)
        
        with torch.no_grad():
            full_query_states, full_key_states = apply_rotary_pos_emb(
                full_query_states, full_key_states, cos, sin, unsqueeze_dim=2
            )
            
            # Repeat k/v heads if num_key_value_heads < num_heads
            if self.num_key_value_groups > 1:
                full_key_states = full_key_states.repeat_interleave(self.num_key_value_groups, dim=2)
                full_value_states = full_value_states.repeat_interleave(self.num_key_value_groups, dim=2)
            
            # Regular attention computation for full attention
            full_key_states = full_key_states.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
            full_value_states = full_value_states.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
            full_query_states = full_query_states.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)

            # Compute attention scores
            attn_weights = torch.matmul(full_query_states, full_key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Add causal mask
            causal_mask = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            full_attn_output = torch.matmul(attn_weights, full_value_states)  # (bsz, num_heads, seq_len, head_dim)
            full_attn_output = full_attn_output.transpose(1, 2)  # (bsz, seq_len, num_heads, head_dim)

        # Local attention computation
        local_query_states, local_key_states = apply_rotary_pos_emb(
            local_query_states, local_key_states, cos, sin, unsqueeze_dim=2
        )
        
        # Repeat k/v heads for local attention
        if self.num_key_value_groups > 1:
            local_key_states = local_key_states.repeat_interleave(self.num_key_value_groups, dim=2)
            local_value_states = local_value_states.repeat_interleave(self.num_key_value_groups, dim=2)
        
        local_key_states = local_key_states.transpose(1, 2)
        local_value_states = local_value_states.transpose(1, 2)
        local_query_states = local_query_states.transpose(1, 2)
        
        # Compute local attention scores with window
        local_attn_weights = torch.matmul(local_query_states, local_key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create local attention mask
        local_mask = torch.ones((q_len, q_len), device=hidden_states.device)
        window_size = min(self.window_size, q_len)
        for i in range(q_len):
            start = max(0, i - window_size + 1)
            local_mask[i, :start] = float('-inf')
        
        # Apply masks
        local_attn_weights = local_attn_weights + local_mask.unsqueeze(0).unsqueeze(0)
        local_attn_weights = torch.softmax(local_attn_weights, dim=-1)
        
        local_attn_output = torch.matmul(local_attn_weights, local_value_states)
        local_attn_output = local_attn_output.transpose(1, 2)

        # Project and reshape outputs
        with torch.no_grad():
            full_attn_output = full_attn_output.reshape(bsz, q_len, self.hidden_size)
            full_attn_output = self.o_proj(full_attn_output)

        local_attn_output = local_attn_output.reshape(bsz, q_len, self.hidden_size)
        local_attn_output = self.o_proj(local_attn_output)

        # Combine outputs
        attn_output = torch.cat([full_attn_output, local_attn_output], dim=0)

        return attn_output, None, past_key_value

class MinimalRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, num_heads=32, max_position_embeddings=2048):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, position_ids=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Create cos and sin of shape [batch_size, seq_len, num_heads, head_dim]
        cos = torch.ones((batch_size, seq_len, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        sin = torch.ones((batch_size, seq_len, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        
        return cos, sin

def replace_attention_with_masking(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
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
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
num_training_steps = len(train_loader) * 4
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

#scaler = GradScaler()
accumulation_steps = 4
model.train()

for epoch in range(4):
    for i, batch in enumerate(train_loader):
        try:
            inputs = {key: val.to(device) for key, val in batch.items()}
            
            with torch.amp.autocast('cuda'):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / accumulation_steps
            
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

        except RuntimeError as e:
            if "out of memory" in str(e):
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
    output_dir = "./trained_model_duo"
else:
    output_dir = "./trained_model_no_mask_duo"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")


