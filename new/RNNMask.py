from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNForgetMask(nn.Module):
    def __init__(self, base_llm_model, threshold=0.5, hidden_dim=512, rnn_type="GRU", bidirectional=True):
        """
        Initializes the RNNForgetMask module.

        Args:
            base_llm_model (nn.Module): The base language model (e.g., a transformer model).
            threshold (float): Threshold for forgetting. Probabilities below this are set to 0 (forgotten).
            hidden_dim (int): Hidden layer size for the RNN.
            rnn_type (str): Type of RNN to use ("RNN", "LSTM", or "GRU").
        """
        super(RNNForgetMask, self).__init__()
        self.embedding = base_llm_model.model.embed_tokens
        self.threshold = threshold

        # Retrieve the embedding dimension from the base model
        embedding_dim = base_llm_model.config.hidden_size

        # Define the RNN layer
        rnn_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_class(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            hidden_dim *= 2

        # Linear layer to output probabilities
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs a probability for retaining each token
        )

    def create_rnn_forgetting_mask(self, tokens):
        """
        Creates a mask for tokens to retain/forget using the RNN.

        Args:
            tokens (torch.Tensor): Token embeddings of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, seq_len),
                          where 1 indicates retained tokens and 0 indicates forgotten tokens.
        """
        batch_size, seq_len, embed_dim = tokens.size()
       # print('dfdfsfdsdfdfsfds')
       # print(tokens.shape)

        # Pass tokens through RNN
        rnn_output, _ = self.rnn(tokens)  # rnn_output: (batch_size, seq_len, hidden_dim)

      #  print(rnn_output.shape)

        # Compute retain probabilities
        retain_probs = self.output_layer(rnn_output).squeeze(-1)  # Shape: (batch_size, seq_len)


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

        # Create mask
        mask = self.create_rnn_forgetting_mask(tokens)  # Shape: (batch_size, seq_len)

        return mask