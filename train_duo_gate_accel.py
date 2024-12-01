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
import torch.nn.functional as F
#import wandb
import os
import json
import dill

mask = True
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

class GatedDuoAttention(nn.Module):
    def __init__(self, base_attention: nn.Module, window_size: int = 128, reduction_factor: int = 4):
        super().__init__()
        # Use base_attention's attributes
        self.hidden_size = base_attention.hidden_size
        self.num_heads = base_attention.num_heads
        self.head_dim = base_attention.head_dim
        self.window_size = window_size

        # Ensure num_heads is divisible by reduction_factor
        assert self.num_heads % reduction_factor == 0
        self.reduction_factor = reduction_factor
        self.reduced_num_heads = self.num_heads // self.reduction_factor

        # Create projections with correct dimensions
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size // self.reduction_factor, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size // self.reduction_factor, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Add gating mechanism
        self.gate_proj = nn.Linear(self.hidden_size, 1)
        self.gate_norm = nn.LayerNorm(self.hidden_size)
        
        self.gate_reg_strength = 0.1
        self.gate_threshold = 0.5

        # Copy weights from base attention
        self.q_proj.weight.data.copy_(base_attention.q_proj.weight.data)
        self.k_proj.weight.data.copy_(base_attention.k_proj.weight.data)
        self.v_proj.weight.data.copy_(base_attention.v_proj.weight.data)
        self.o_proj.weight.data.copy_(base_attention.o_proj.weight.data)

        # Pre-compute masks for maximum sequence length (usually 2048 for Llama)
        max_seq_len = 2048  # or whatever your maximum sequence length is
        self.max_seq_len = max_seq_len
        
        # Register masks as buffers so they're automatically moved to the right device
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(1, 1, max_seq_len, max_seq_len) * float("-inf"),
                diagonal=1
            )
        )
        
        self.register_buffer(
            "local_window_mask",
            self._create_local_window_mask(max_seq_len)
        )

    def _create_local_window_mask(self, seq_len: int) -> torch.Tensor:
        local_window_mask = torch.ones(1, 1, seq_len, seq_len) * float("-inf")
        window_size = min(seq_len, self.window_size)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            local_window_mask[:, :, i, start:end] = 0
        return local_window_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        # Compute gates with gradient clipping
        gate_input = self.gate_norm(hidden_states)
        gates = torch.sigmoid(self.gate_proj(gate_input).clamp(-10, 10))  # Add clipping

        # Project states
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.reduced_num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.reduced_num_heads, self.head_dim).transpose(1, 2)

        # Repeat K and V to match number of heads
        key_states = key_states.repeat(1, self.reduction_factor, 1, 1)
        value_states = value_states.repeat(1, self.reduction_factor, 1, 1)

        # Apply gating - reshape gates to match attention dimensions
        gate_binary = (gates <= self.gate_threshold).float()
        gate_binary = gate_binary.squeeze(-1).unsqueeze(1).unsqueeze(-1)
        gate_binary = gate_binary.expand(bsz, self.num_heads, q_len, q_len)
        #breakpoint()

        # Use pre-computed masks and slice to current sequence length
        causal_mask = self.causal_mask[:, :, :q_len, :q_len].expand(bsz, self.num_heads, -1, -1)
        local_window_mask = self.local_window_mask[:, :, :q_len, :q_len].expand(bsz, self.num_heads, -1, -1)

        # Combine masks
        combined_mask = torch.where(gate_binary == 1, local_window_mask, causal_mask)
        

        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + combined_mask

        # Use stable softmax
        attn_probs = F.softmax(scores.float(), dim=-1, dtype=torch.float32).to(scores.dtype)
        
        # Apply attention
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Final projection
        output = self.o_proj(attn_output)

        # Compute regularization losses
        eps = 1e-5
        gates_stable = gates.clamp(eps, 1-eps)
        
        # Binary regularization to prevent gates from being exactly 0 or 1
        binary_reg = self.gate_reg_strength * torch.mean(gates * (1 - gates))
        
        # Entropy regularization for uncertainty
        entropy_reg = -0.01 * torch.mean(
            gates * torch.log(gates_stable) + 
            (1 - gates) * torch.log(1 - gates_stable)
        )
        
        # L1 regularization to encourage sparsity (smaller gates)
        sparsity_reg = 0.1 * torch.mean(gates)  # Added this term to push gates toward 0
        
        # Combined loss that prefers local attention
        reg_loss = binary_reg + entropy_reg + sparsity_reg

        # Store statistics
        self.last_gate_mean = gates.mean().item()
        self.last_gate_std = gates.std().item()
        self.last_reg_loss = reg_loss.item()
        
        return output, reg_loss, past_key_value
    
def create_causal_mask(batch_size: int, num_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(batch_size, num_heads, seq_len, seq_len, device=device) * float("-inf"),
        diagonal=1
    )

def create_local_window_mask(batch_size: int, num_heads: int, window_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    local_window_mask = torch.ones(batch_size, num_heads, seq_len, seq_len, device=device) * float("-inf")
    window_size = min(seq_len, window_size)  # Use smaller of window size or sequence length
    
    # Fill in the window
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        local_window_mask[:, :, i, start:end] = 0
    return local_window_mask

def replace_attention_with_masking(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            print(f"Replacing attention module: {name}")
            hierarchical_attention = GatedDuoAttention(module)
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

if __name__ == "__main__":
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

    # Initialize wandb if you want to use it
    #wandb.init(project="llama-gated-attention")

    if mask:
        output_dir = "./trained_model_duo_gate"
    else:
        output_dir = "./trained_model_no_mask_duo_gate"



    for epoch in range(4):
        for i, batch in enumerate(train_loader):
            try:
                inputs = {key: val.to(device) for key, val in batch.items()}
                
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    main_loss = outputs.loss / accumulation_steps
                    
                    # Collect regularization losses from all attention layers
                    reg_loss = 0
                    for name, module in model.named_modules():
                        if isinstance(module, GatedDuoAttention):
                            # Get the reg_loss from the last forward pass
                            reg_loss += module.last_reg_loss
                    
                    # Combine losses
                    total_loss = main_loss + reg_loss

                total_loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if i % 10 == 0:
                    # Log metrics
                    print(f"Epoch {epoch}, Step {i}")
                    print(f"Main Loss: {main_loss.item():.4f}")
                    print(f"Reg Loss: {reg_loss:.4f}")
                    
                    # Log gate statistics from first attention layer
                    for name, module in model.named_modules():
                        if isinstance(module, GatedDuoAttention):
                            print(f"Layer {name}")
                            print(f"Gate Mean: {module.last_gate_mean:.3f}")
                            print(f"Gate Std: {module.last_gate_std:.3f}")
                            
                            # Log to wandb if enabled
                            
                            break  # Just log first layer for brevity

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


    # At the end of training:
    if mask:
        output_dir = "./trained_model_duo_gate_new"
    else:
        output_dir = "./trained_model_no_mask_duo_gate_new"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model config
    model_config = {
        'mask': mask,
        'window_size': 128,
        'reduction_factor': 4,
        'base_model_name': "meta-llama/Llama-3.2-1B",
    }
    
    # Save config as JSON
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save model state dict separately
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "pytorch_model.pt"),
        pickle_module=dill,
    )

    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Model, config, and tokenizer saved to {output_dir}")


