

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class NER_SA_Dataset(Dataset):
    """
    Custom dataset for multitask NLP tasks: Named Entity Recognition (NER) + Sentiment Analysis (SA).

    Returns:
        - List of tokens per sentence (for CharEmbedding).
        - Encoded NER labels (one per token).
        - Encoded sentiment label (one per sentence).
    """

    def __init__(
        self,
        texts: List[List[str]],
        ner_tags: List[List[str]],
        sentiments: List[str],
        label_map: Dict[str, int],
        sentiment_map: Dict[str, int],
    ) -> None:
        """
        Initializes the NER_SA_Dataset instance.

        Args:
            texts (List[List[str]]): List of tokenized sentences.
            ner_tags (List[List[str]]): List of NER tags for each token in each sentence.
            sentiments (List[str]): List of sentiment labels for each sentence.
            label_map (dict): Dictionary mapping NER tags to numeric labels.
            sentiment_map (dict): Dictionary mapping sentiment labels to numeric values.
        """
        
        assert len(texts) == len(ner_tags) == len(sentiments), "Tamaños desalineados"

        self.texts = texts
        
        # Create the NER labels by mapping the string labels to integer indices using label_map
        self.targets_ner: List[List[int]] = [
            [label_map[label] for label in sentence] for sentence in ner_tags
        ]
        
        # Map sentiment labels to their integer representations
        self.targets_sa: List[int] = [sentiment_map[sent] for sent in sentiments]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Retrieves the input tokens, NER labels, and sentiment label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[List[str], torch.Tensor, torch.Tensor]: 
                - List of tokens for the sentence.
                - Tensor of NER labels for each token in the sentence.
                - Tensor of sentiment label for the sentence.
        """
        
        tokens: List[str] = self.texts[idx]
        target_ner: torch.Tensor = torch.tensor(self.targets_ner[idx], dtype=torch.long)
        target_sa: torch.Tensor = torch.tensor(self.targets_sa[idx], dtype=torch.long)
        return tokens, target_ner, target_sa


class AlertDataset(Dataset):
    """
    Custom PyTorch Dataset for fine-tuning a T5 model to generate alerts.

    """
    
    def __init__(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> None:
        """
        Initializes the dataset with input and output tokenized sequences.

        Args:
            inputs (Dict): Dictionary containing 'input_ids' and 'attention_mask'.
            outputs (Dict): Dictionary containing 'input_ids' for target sequences, used as labels.
        """
        
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx: int)-> Dict[str, torch.Tensor]:
        """
        Retrieves the input and output tensors for a given index.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            A dictionary with:
                - 'input_ids': Token IDs of the input sequence.
                - 'attention_mask': Mask for padding tokens.
                - 'labels': Token IDs of the target sequence.
        """
        
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.outputs['input_ids'][idx]
        }
