from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMLPForgetMask(nn.Module):
    def __init__(self, base_llm_model, hidden_dim=512, cnn_hidden_dim=256, kernel_size=3):
        """
        Initializes the CNNMLPForgetMask module.

        Args:
            base_llm_model (nn.Module): The base language model (e.g., a transformer model).
            threshold (float): Threshold for forgetting. Probabilities below this are set to 0 (forgotten).
            hidden_dim (int): Hidden layer size for the MLP.
            cnn_hidden_dim (int): Number of output channels for the CNN.
            kernel_size (int): Size of the convolutional kernel.
        """
        super(CNNMLPForgetMask, self).__init__()
        self.embedding = base_llm_model.model.embed_tokens

        # Retrieve the embedding dimension from the base model
        embedding_dim = base_llm_model.config.hidden_size

        # Define the CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_hidden_dim, out_channels=embedding_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs a probability for retaining each token
        )

    def create_cnn_mlp_forgetting_mask(self, tokens):
        """
        Creates a mask for tokens to retain/forget using CNN + MLP.

        Args:
            tokens (torch.Tensor): Token embeddings of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, seq_len),
                          where 1 indicates retained tokens and 0 indicates forgotten tokens.
        """
        batch_size, seq_len, embed_dim = tokens.size()

        # Apply CNN: Convert to (batch_size, embed_dim, seq_len) for Conv1d
        tokens_cnn_input = tokens.transpose(1, 2)  # Shape: (batch_size, embed_dim, seq_len)
        tokens_cnn_output = self.cnn(tokens_cnn_input)  # Shape: (batch_size, embed_dim, seq_len)
        tokens_processed = tokens_cnn_output.transpose(1, 2)  # Shape: (batch_size, seq_len, embed_dim)

        # Flatten token embeddings for processing through MLP
        tokens_flat = tokens_processed.view(-1, embed_dim)  # Shape: (batch_size * seq_len, embed_dim)

        # Compute retain probabilities
        retain_probs = self.mlp(tokens_flat).view(batch_size, seq_len)  # Shape: (batch_size, seq_len)


        return retain_probs

    def forward(self, input_ids):
        """
        Generates the forgetting mask and applies it to the token embeddings.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Masked token embeddings.
        """
        # Get token embeddings
        tokens = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embedding_dim)

        # Create mask using CNN + MLP
        mask = self.create_cnn_mlp_forgetting_mask(tokens)  # Shape: (batch_size, seq_len)

        return mask