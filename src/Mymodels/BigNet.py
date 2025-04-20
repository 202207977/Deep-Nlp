import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Tuple

from flair.embeddings import CharacterEmbeddings
from flair.data import Sentence


class PretrainedCharEmbedder(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the PretrainedCharEmbedder using Flair CharacterEmbeddings.

        Returns:
            None
        """
        
        super().__init__()
        
        # Load Flair character-level embeddings (you can replace with CharLMEmbeddings, etc.)
        self.char_embedder = (
            CharacterEmbeddings()
        ) 

    def forward(self, token_list_batch: List[List[str]]) -> torch.Tensor:
        """
        Forward pass to generate character-based embeddings.

        Args:
            token_list_batch: A list of tokenized sentences, where each sentence is a list of tokens (strings).

        Returns:
            A padded tensor of shape (batch_size, max_seq_len, embedding_dim) containing character embeddings.
        """
        
        embeddings: List[torch.Tensor] = []
        for tokens in token_list_batch:
            # Convert tokens into Flair's Sentence format
            sentence: Sentence = Sentence(tokens)
            
            # Embed sentence tokens using the pretrained character-level model
            self.char_embedder.embed(sentence)
            
            # Stack token embeddings into a tensor
            token_embeddings: torch.Tensor = torch.stack(
                [token.embedding for token in sentence], dim=0
            )

            embeddings.append(token_embeddings)

        # Determine maximum sequence length in the batch
        max_len: int = max(e.size(0) for e in embeddings)
        batch_size: int = len(embeddings)
        emb_dim: int = embeddings[0].size(1)
        
        # Pad each sequence to the maximum length
        padded: torch.Tensor = torch.zeros(batch_size, max_len, emb_dim)

        for i, emb in enumerate(embeddings):
            padded[i, : emb.size(0), :] = emb

        return padded  # (batch, seq_len, emb_dim)


class BigNet(nn.Module):
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
        Initializes the BigNet architecture for NER + Sentiment tasks.

        Args:
            word_vocab_size: Size of the vocabulary.
            word_emb_dim: Dimension of word embeddings.
            pretrained_word_embeddings: Pretrained word vectors (frozen or trainable).
            lstm_hidden_dim: LSTM hidden dimension.
            ner_num_classes: Number of NER classes.
            sentiment_num_classes: Number of sentiment classes.
            char_emb_dim: Output dimension of Flair character embeddings.
            label_map: Mapping from NER labels to indices.
            char2idx: Character-to-index dictionary (not directly used here).

        Returns:
            None
        """
        
        super().__init__()

        # Character embedding module
        self.char_embedding: PretrainedCharEmbedder = PretrainedCharEmbedder()

        # Word2Vec embedding
        self.word_embedding: nn.Embedding = nn.Embedding.from_pretrained(
            pretrained_word_embeddings, freeze=False
        )

        # Word LSTM
        self.lstm_word: nn.LSTM = nn.LSTM(
            word_emb_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        )

        # NER classifier
        self.ner_classifier: nn.Linear = nn.Linear(2 * lstm_hidden_dim, ner_num_classes)

        # Project NER output to char_emb_dim
        self.ner_char_proj: nn.Linear = nn.Linear(ner_num_classes, char_emb_dim)

        # Final LSTM after char + ner-char embeddings (es una deep LSTM)
        self.final_lstm: nn.LSTM = nn.LSTM(
            input_size=2 * char_emb_dim,    
            hidden_size=lstm_hidden_dim,    
            num_layers=3,                   
            batch_first=True,
            bidirectional=False             
        )

        # Sentiment classification
        self.sentiment_classifier: nn.Linear = nn.Linear(
            lstm_hidden_dim, sentiment_num_classes
        )
        self.label_map: Dict[str, int] = label_map 
        self.idx2label: Dict[int, str] = {v: k for k, v in label_map.items()}


    def forward(self, word_input: torch.Tensor, token_list_batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the BigNet architecture.

        Args:
            word_input: Tensor of shape (batch_size, seq_len) with token indices.
            token_list_batch: List of tokenized words per input sample (used for character-level embedding).

        Returns:
            Tuple containing:
                - Sentiment output: Tensor of shape (batch_size, sentiment_num_classes)
                - NER output: Tensor of shape (batch_size, seq_len, ner_num_classes)
        """
        
        # === NER Path ===
        # Get word-level embeddings for each token in the input sequence
        word_emb: torch.Tensor = self.word_embedding(word_input)  # (batch, seq_len, emb_dim)

        # Pass word embeddings through a BiLSTM to capture contextual word representations
        lstm_out, _ = self.lstm_word(word_emb)  # (batch, seq_len, 2*hidden)

        # Apply a linear layer to get NER class scores for each token
        ner_out: torch.Tensor = self.ner_classifier(lstm_out)  # (batch, seq_len, ner_classes)

        # Choose the most probable class for each token
        ner_preds: torch.Tensor = ner_out.argmax(dim=-1)  # (batch, seq_len, word_size)

        # === Concatenation of Tokens and NER Tags ===
        
        # Convert predicted indices to NER tag strings (batch , n_letras_frase+ n_letras_ner)
        ner_token_texts: List[List[str]] = [
            [self.idx2label[class_id.item()] for class_id in sentence]
            for sentence in ner_preds
        ]

        # Concatenate original tokens with predicted NER labels
        concat_tokens: List[List[str]] = []
        for i in range(len(token_list_batch)):
            concat_tokens.append(token_list_batch[i] + ner_token_texts[i])
            
        # === Character Embedding Path ===
        
        # Compute character-level embeddings of the concatenated token + NER-tag sequences
        char_emb: torch.Tensor = self.char_embedding(concat_tokens)  # (batch, seq_len, emb_dim)
        char_emb = char_emb.to(word_input.device)

        # === Final LSTM + Sentiment classifier ===
        
        # Pass the character-level embeddings through a stacked LSTM for deeper representation
        final_lstm_out, _ = self.final_lstm(char_emb)  # (batch, seq_len, 2*hidden)
        
        # Average the LSTM outputs over the sequence length (mean pooling)
        pooled = torch.mean(final_lstm_out, dim=1)  # (batch, 2*hidden)
        
        sentiment_output: torch.Tensor = self.sentiment_classifier(pooled) # (batch, sentiment_classes)

        return sentiment_output, ner_out
        
