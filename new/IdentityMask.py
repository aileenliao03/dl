from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityMask(nn.Module):
  def __init__(self):
      super(IdentityMask, self).__init__()

  def forward(self, input_ids):
      return torch.ones_like(input_ids, dtype=torch.bool)
