from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomForgetMask(nn.Module):
  def __init__(self, forget_prob = .1):
      super(RandomForgetMask, self).__init__()
      self.forget_prob = forget_prob

  def create_random_forgetting_mask(self, input_ids):
      """
      Creates a mask for tokens to forget based on the forget rate.

      Args:
          input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len).

      Returns:
          torch.Tensor: Binary mask of the same shape as input_ids,
                        where 1 indicates retained tokens and 0 indicates forgotten tokens.
      """
      batch_size, seq_len = input_ids.size()
      num_tokens_to_forget = int((seq_len) * self.forget_prob)
      masks = []

      for i in range(batch_size):
          # Generate random indices for tokens to forget
          forget_indices = torch.randperm(seq_len)[:num_tokens_to_forget]
          mask = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
          mask[forget_indices] = False  # Mark forgotten tokens as 0
          masks.append(mask)

      return torch.stack(masks)

  def forward(self, input_ids):
      mask = self.create_random_forgetting_mask(input_ids)
      return mask
