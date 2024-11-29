from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import wandb
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset, Dataset
import shutil
from time import sleep
from datasets import load_dataset, DatasetDict, Dataset
login(token="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")


wandb.login(key="a1df0c88c10ee7966670142e4afd5a895138c7cf")
device =  ('cuda' if torch.cuda.is_available() else 'cpu')
from Adaptive import AttentionForgetMask
from RNNMask import RNNForgetMask
from RandomForgetMask import RandomForgetMask
from IdentityMask import IdentityMask
from MLPForgetMask import MLPForgetMask
from CNNMask import CNNMLPForgetMask
from helper import apply_mask, memory_check_and_empty

def train_model(base_model, forget_layer,soft, train_dataloader, optimizer, scheduler, num_epochs, device, model_name, save_dir, batch_size=64):

    wandb.init(project="deeplearning",  # Use your desired project name here
    entity="ai0liao",config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "model_name": model_name,
        "dataset": "wikipedia",
    })
    
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    accumulation_steps = batch_size // train_dataloader.batch_size  # Gradient accumulation steps
    total_steps = num_epochs * len(train_dataloader) // accumulation_steps
    save_interval = (total_steps // 3) +1
    step_count = 0
    progress_bar = tqdm(total=total_steps, desc="Training")

    # Initialize mixed precision scaler
    scaler = GradScaler()

    base_model.train()
    forget_layer.train()
    for epoch in range(num_epochs):
        epoch_loss = 0  # Reset loss for each epoch
        for batch_idx, batch in enumerate(train_dataloader):
          input_ids = batch["input_ids"].to(device)
       #   attention_mask = batch["attention_mask"].to(device)
          mask = forget_layer(input_ids)

          if soft: #soft mask
            embeddings = base_model.model.embed_tokens(input_ids)
            new_mask = mask.unsqueeze(-1).expand_as(embeddings)
            new_input_embeddings = new_mask*embeddings
    #        attention_mask = (mask*attention_mask).long()
            labels = input_ids
          else:#hard mask remove values
            new_input_ids = apply_mask(input_ids, mask)
      #      attention_mask = apply_mask(attention_mask, mask)

            #Truncate attention_mask and labels token num is same
            batchsize, seq_len = new_input_ids.size()
            labels = input_ids[:, -1*seq_len:]

          with autocast():
            if soft:
              outputs = base_model(inputs_embeds=new_input_embeddings, labels=labels)
            else:
              outputs = base_llm_model(input_ids=new_input_ids,  labels=labels)#attention_mask=attention_mask,
            loss = outputs.loss / accumulation_steps
            if soft:
              #mask loss
              lambda_l1 =1
              mask_loss =  lambda_l1 * torch.mean(torch.abs(mask))

          if soft:
            mloss = mask_loss + loss
            scaler.scale(mloss).backward()
          else:
            scaler.scale(loss).backward()

          if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
              # Gradient update
              scaler.step(optimizer)
              scheduler.step()

              scaler.update()

              optimizer.zero_grad()

              step_count += 1
              progress_bar.update(1)
              logs = {"step_loss": loss.item() * accumulation_steps, "step": step_count}
              if soft:
                logs["mask_loss"] = mask_loss.item() *accumulation_steps
              wandb.log(logs)

          epoch_loss += loss.item() * accumulation_steps  # Accumulate scaled loss
          memory_check_and_empty()
        # Epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})

    # Final model save
    save_path_forget = os.path.join(save_dir, f"forget.pt")
    base_model.save_pretrained(save_dir)
    torch.save(forget_layer, save_path_forget)
    print(f"Model saved")
    wandb.finish()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
torch.backends.cudnn.benchmark = True
batch_size = 4
EXAMPLES = 10000
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_data = dataset["train"].select(range(EXAMPLES))
# Initialize tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to preprocess examples
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = train_data.map(preprocess_function, batched=True, remove_columns=['text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids'])

train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size, num_workers=min(4, torch.get_num_threads()),
        pin_memory=True,
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
#mask = IdentityMask().to(device)
#mask = RandomForgetMask(forget_prob=.1).to(device)
#mask = MLPForgetMask(base_llm_model).to(device)
# mask = RNNForgetMask(base_llm_model, bidirectional=False).to(device)
mask = AttentionForgetMask(base_llm_model).to(device)

soft = False
parameters = list(base_llm_model.parameters()) + list(mask.parameters())
name = "F_GRU_TFTFF2step_10000_Schedule_Reinforce_Step_MLPForget_Full_Fintune"
num_epochs = 5

# Optimizer and Scheduler
optimizer = AdamW(parameters, lr=5e-5)
total_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

train_model(base_llm_model, mask,soft,  train_dataloader,  optimizer, scheduler, num_epochs, device, name, "/home/aileen/dl/new/models")