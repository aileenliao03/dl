import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionForgetMask(nn.Module):
    def __init__(self, base_llm_model, threshold=0.5, retain_percentage=0.9):
        """
        Initializes the AttentionForgetMask module.

        Args:
            base_llm_model (nn.Module): The base language model (e.g., a transformer model).
            threshold (float): Threshold for forgetting. Probabilities below this are set to 0 (forgotten).
            retain_percentage (float): Percentage of tokens to retain based on attention scores (default 0.9).
        """
        super(AttentionForgetMask, self).__init__()
        self.embedding = base_llm_model.get_input_embeddings()  # Access the input embeddings of the Llama model
        self.threshold = threshold
        self.retain_percentage = retain_percentage

    def create_attention_forgetting_mask(self, tokens):
        """
        Creates a mask for tokens to retain/forget based on embedding values.

        Args:
            tokens (torch.Tensor): Token embeddings of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, seq_len),
                          where 1 indicates retained tokens and 0 indicates forgotten tokens.
        """
        batch_size, seq_len, embed_dim = tokens.size()

        # Compute a "relevance" score for each token based on its embedding
        relevance_scores = tokens.norm(dim=-1)  # Compute L2 norm of the embeddings (length of each token vector)
        
        # Rank tokens by relevance, keeping the top `retain_percentage` tokens
        retain_count = int(seq_len * self.retain_percentage)
        _, sorted_indices = torch.topk(relevance_scores, retain_count, dim=1, largest=True, sorted=False)

        # Create binary mask, setting the top tokens to 1 and others to 0
        mask = torch.zeros(batch_size, seq_len, device=tokens.device)
        mask.scatter_(1, sorted_indices, 1)

        return mask

    def forward(self, input_ids):
        """
        Generates the forgetting mask based on token embeddings.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, seq_len),
                          where 1 indicates retained tokens and 0 indicates forgotten tokens.
        """
        # Get token embeddings from the base model
        tokens = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embedding_dim)

        # Create the forgetting mask based on embeddings
        mask = self.create_attention_forgetting_mask(tokens)  # Shape: (batch_size, seq_len)

        return mask
