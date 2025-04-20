from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flair.embeddings import CharacterEmbeddings
from flair.data import Sentence


class SmallNet(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        word_emb_dim: int,
        pretrained_word_embeddings: torch.Tensor,
        lstm_hidden_dim: int,
        ner_num_classes: int,
        sentiment_num_classes: int,
        char_emb_dim: int,
        label_map: Dict[str, int],
        char2idx: Dict[str, int]
    ) -> None:  # char_emb_dim = output dim of flair embeddings 
        
        """
        Initializes the SmallNet class for NER and Sentiment Analysis.

        Args:
            word_vocab_size: Total vocabulary size.
            word_emb_dim: Dimensionality of word embeddings.
            pretrained_word_embeddings: Pretrained word embedding matrix.
            lstm_hidden_dim: Hidden size for LSTM layers.
            ner_num_classes: Number of output classes for the NER task.
            sentiment_num_classes: Number of sentiment classification labels.
            char_emb_dim: Dimensionality of the character-level embeddings.
            label_map: Dictionary mapping NER label names to indices.
            char2idx: Dictionary mapping characters to indices (not used directly here).

        Returns:
            None
        """
        super().__init__()


        # Word2Vec embedding
        self.word_embedding: nn.Embedding = nn.Embedding.from_pretrained(
            pretrained_word_embeddings, freeze=False
        )

        # BiLSTM for Named Entity Recognition
        self.lstm_nerd: nn.LSTM = nn.LSTM(
            word_emb_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        )

        # NER classifier
        self.ner_classifier: nn.Linear = nn.Linear(2 * lstm_hidden_dim, ner_num_classes)
        
        # 3-layer LSTM for Sentiment Analysis (non-bidirectional)
        self.lstm_sa= nn.LSTM(
            input_size=word_emb_dim,  
            hidden_size=lstm_hidden_dim,    
            num_layers=3,                   
            batch_first=True,
            bidirectional=False             
        )

        # SA classifier
        self.sentiment_classifier: nn.Linear = nn.Linear(
            lstm_hidden_dim, sentiment_num_classes
        )
        
        # Dropout to reduce overfitting (used in sentiment classification)
        self.dropout: nn.Dropout = nn.Dropout(p=0.5)
        
        # Mapping of labels to indices and inverse
        self.label_map: Dict[str, int] = label_map 
        self.idx2label: Dict[int, str] = {v: k for k, v in label_map.items()}
        
        


    def forward(self, word_input: torch.Tensor, token_list_batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SmallNet model.

        Args:
            word_input: Tensor of shape (batch_size, seq_len), containing token indices.
            token_list_batch: List of tokenized sentences for potential use with char embeddings.

        Returns:
            Tuple:
                - Sentiment output tensor of shape (batch_size, sentiment_num_classes)
                - NER output tensor of shape (batch_size, seq_len, ner_num_classes)
        """
        
        # === NER Path ===
        # Convert token indices into word embeddings
        word_emb: torch.Tensor = self.word_embedding(word_input)  # (batch, seq_len, emb_dim)

        # Pass word embeddings through BiLSTM for NER
        ner_lstm_out, _ = self.lstm_nerd(word_emb)  # (batch_size, seq_len, 2*lstm_hidden_dim)

        # Predict NER labels for each token
        ner_out: torch.Tensor = self.ner_classifier(ner_lstm_out)  # (batch, seq_len, ner_classes)


        # === SA Path ===
        # Pass same word embeddings through a separate LSTM for sentiment
        sa_lstm_out, _ = self.lstm_sa(word_emb)  # (batch, seq_len, 2*hidden)
        
        # Apply mean pooling over the sequence
        pooled: torch.Tensor = torch.mean(sa_lstm_out, dim=1)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Predict sentiment for the full sequence
        sa_out: torch.Tensor = self.sentiment_classifier(pooled) # (batch, sentiment_num_classes)

        return sa_out, ner_out
        
