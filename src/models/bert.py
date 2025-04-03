import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertNet(nn.Module):
    """ """

    def __init__(self) -> None:
        super(BertNet, self).__init__()
        """
        Constructor of the Bert class.
        """

        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs: str, the input text.

        Return:
            A tensor, with the embedded sentence, of shape [sequence len, embedding dim]
        """

        tokens = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**tokens)

        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states  # (batch_size, sequence_length, embedding_dim)
