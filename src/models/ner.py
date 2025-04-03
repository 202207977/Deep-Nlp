import torch.nn as nn
from bert import BertNet


class NERNet(nn.Module):
    """
    Modelo de Named Entity Recognition (NER) usando BERT y BiLSTM.
    """

    def __init__(self, num_labels: int, dropout_rate: float = 0.3):
        """
        Args:
            num_labels (int): Número de etiquetas NER.
            dropout_rate (float): Probabilidad de dropout para regularización.
        """
        super(NERNet, self).__init__()

        self.bert_net = BertNet()
        self.lstm = nn.LSTM(
            self.bert_net.model.config.hidden_size,
            256,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256 * 2, num_labels)  # hidden_dim * 2 por BiLSTM

    def forward(self, inputs):
        """
        Forward pass del modelo.

        Args:
            inputs (str o torch.Tensor): Entrada de texto (str) o tensor de tokens (batch_size, seq_length).

        Returns:
            torch.Tensor: Logits de tamaño (batch_size, seq_length, num_labels).
        """

        bert_embeddings = self.bert_net(
            inputs
        )  # (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(
            bert_embeddings
        )  # (batch_size, seq_length, hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (batch_size, seq_length, num_labels)

        return logits
