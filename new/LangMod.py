from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

class Language_Model(nn.Module):
  def __init__(self, base_llm_model, forget_layer, soft=False, choice_config=None ):
    super().__init__()
    self.base_llm_model = base_llm_model
    self.forget_layer = forget_layer
    self.soft=soft
    self.choice_config = choice_config

  def apply_mask(self, input_ids, mask, padding_value=0, attention =False):
        """
        Applies a forgetting mask to input_ids to retain only unmasked tokens and pads sequences to uniform length.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len).
            mask (torch.Tensor): Binary mask of the same shape as input_ids,
                                where 1 indicates retained tokens and 0 indicates forgotten tokens.
            padding_value (int, optional): Value to pad the sequences to uniform length. Defaults to 0.

        Returns:
            torch.Tensor: Padded tensor of shape (batch_size, max_len), where max_len is the length of the longest retained sequence.
        """
        if not attention:
          if self.soft:
            embeddings = self.base_model.model.embed_tokens(input_ids)
            mask = mask.unsqueeze(-1).expand_as(embeddings)
            new_input_embeddings = mask*embeddings
            return {"embeddings": new_input_embeddings}

          #mask (batch , seq)
          if self.choice_config !=None:
            if "percent" in self.choice_config: # top x percent
              percent = self.choice_config["percent"]
              k = int(mask.size(1) * percent)
              _, indices = torch.topk(mask, k, dim=1)
              new_mask = torch.zeros_like(mask)
              new_mask.scatter_(1, indices, 1)          # Set top-k values to 1
              mask = new_mask
            elif "threshold" in self.choice_config: #above threshold
              threshold = self.choice_config["threshold"]
              mask = (mask > threshold).float()
            count = torch.sum(mask == 0).item()

        updated_input_ids = []

        for i in range(input_ids.size(0)):
            # Use the mask to retain tokens from input_ids
            retained_input_ids = input_ids[i][mask[i].bool()]
            updated_input_ids.append(retained_input_ids)

        # Find the maximum sequence length after masking
        max_length = max(ids.size(0) for ids in updated_input_ids)

        if max_length == 0:
            #only return last tokens
          return input_ids[:, -1].rehsape(-1, 1)


          # Pad sequences to the maximum length
        padded_input_ids = torch.full(
              (input_ids.size(0), max_length),
              padding_value,
              dtype=input_ids.dtype,
              device=input_ids.device
          )

        for i, ids in enumerate(updated_input_ids):
            padded_input_ids[i, :ids.size(0)] = ids

        return {"input_ids": padded_input_ids}#, "count":count}


  def forward(self, input_ids, attention_mask=None, labels = None):
    mask = self.forget_layer(input_ids)
    data = self.apply_mask(input_ids, mask)
    if "input_ids" in data:
      input_ids = data["input_ids"]
      batch, seq_len = input_ids.size()
    else:
      input_embedings = data["embeddings"]
      batch, seq_len, _ = input_embedings.size()


    if attention_mask != None:
      attention_mask = self.apply_mask(attention_mask, mask)["input_ids"]

    if labels != None:
      labels = labels[:, -1*seq_len:]

    if "input_ids" in data:
      output = self.base_llm_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels )
    elif "embeddings" in data:
      output = self.base_llm_model(inputs_embeds=input_embedings, attention_mask=attention_mask, labels=labels )

    logits = output.logits  # Assuming 'output' is a named tuple or a dictionary

    return_data = {"logits": logits, "seq_len": input_ids.size(1), "loss": output.loss}
    if "count" in data:
      return_data["count"] = data["count"]
    return return_data


  def generate(self, input_ids, attention_mask =None, max_length=50):
      # Generate text by iteratively sampling next tokens
      for _ in range(max_length):
          # Run input through the model to get next token probabilities
          outputs = self.forward(input_ids,attention_mask = attention_mask)
          next_token_logits = outputs["logits"][:, -1, :]  # Only use the logits for the last token

          # Sample the next token
          next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

          if attention_mask != None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

          # Append the new token to the sequence
          input_ids = torch.cat([input_ids, next_token], dim=-1)

      return input_ids