import torch

def apply_mask( input_ids, mask, padding_value=0):
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
      updated_input_ids = []

      for i in range(input_ids.size(0)):
          # Use the mask to retain tokens from input_ids
          retained_input_ids = input_ids[i][mask[i].bool()]
          updated_input_ids.append(retained_input_ids)



      # Find the maximum sequence length after masking
      max_length = max(ids.size(0) for ids in updated_input_ids)

      if max_length == 0:
        #only return last tokens
        return input_ids[:, -1]


        # Pad sequences to the maximum length
      padded_input_ids = torch.full(
            (input_ids.size(0), max_length),
            padding_value,
            dtype=input_ids.dtype,
            device=input_ids.device
        )

      for i, ids in enumerate(updated_input_ids):
          padded_input_ids[i, :ids.size(0)] = ids

      return padded_input_ids
def memory_check_and_empty():
    """Check GPU memory and clear cache only if necessary."""
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        if reserved_memory - allocated_memory > 0.1 * reserved_memory:  # Threshold for unused memory
            torch.cuda.empty_cache()