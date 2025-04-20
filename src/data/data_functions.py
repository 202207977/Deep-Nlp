from typing import List, Dict, Tuple, Set
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from collections import Counter
import string

from torch.nn.utils.rnn import pad_sequence

# Own libraries
from src.data.dataloader import NER_SA_Dataset

DATASET = "data/dataset.csv"

def build_word2idx(texts: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """
    Builds a word-to-index dictionary based on frequency threshold.

    Args:
        texts (List[List[str]]): List of tokenized sentences.
        min_freq (int): Minimum frequency to include a word in the vocabulary.

    Returns:
        Dict[str, int]: Dictionary mapping words to indices.
    """
    
    counter: Counter = Counter(token.lower() for sent in texts for token in sent)
    
    # Initialize vocabulary with special tokens for padding and unknown words
    word2idx: Dict[str, int] = {
        "<pad>": 0,  # Used for padding sequences to the same length
        "<unk>": 1   # Used for words not present in the vocabulary
    }
    
    # Add words that meet the minimum frequency requirement
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx) # Assign the next available index
    
    return word2idx

def build_tag2idx(tags: List[List[str]]) -> Dict[str, int]:
    """
    Builds a tag-to-index dictionary for Named Entity Recognition (NER).

    Args:
        tags: A list of tag sequences, where each sequence is a list of NER tags.

    Returns:
        A dictionary mapping each unique NER tag to a unique integer index.
    """
    
    # Extract unique tags from all sentences and sort them for consistency
    all_tags = sorted({tag for sent in tags for tag in sent})
    
    # Assign an index to each tag
    return {tag: i for i, tag in enumerate(all_tags)}

def build_char2idx()-> Dict[str, int]:
    """
    Builds a character-to-index dictionary including standard, accented,
    and special characters, along with padding and unknown tokens.

    Returns:
        A dictionary mapping each character to a unique integer index.
        Special tokens:
            - "<pad>": Padding token, mapped to index 0.
            - "<unk>": Unknown character token, mapped to index 1.
    """

    VOCABULARY: Set[str] = set()

    # Add ASCII letters (a-z, A-Z)
    VOCABULARY.update(string.ascii_letters)

    # Add digits (0-9)
    VOCABULARY.update(string.digits)

    # Add standard punctuation marks
    VOCABULARY.update(string.punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    # Add space
    VOCABULARY.add(' ')

    # Add commonly used accented characters
    VOCABULARY.update("áéíóúüñÁÉÍÓÚÜÑçÇ")

    # Add additional useful symbols (currency, copyright, etc.)
    VOCABULARY.update(["€", "$", "£", "¥", "©", "®", "°", "¿", "¡", "→", "•", "−", "–", "—"])

    # Sort characters to ensure consistent indexing
    VOCABULARY = sorted(VOCABULARY)
    
    # Assign indices starting from 2 (0 for pad, 1 for unk)
    char2idx = {char: idx + 2 for idx, char in enumerate(VOCABULARY)}
    char2idx["<pad>"] = 0
    char2idx["<unk>"] = 1

    return char2idx


def build_sentiment2idx(sentiments: List[str]) -> Dict[str, int]:
    """
    Builds a sentiment-to-index dictionary from the list of sentiment labels.

    Args:
        sentiments: List of sentiment labels

    Returns:
        A dictionary mapping each unique sentiment label to a unique integer index.
    """
    # Get unique sentiment labels sorted alphabetically
    unique = sorted(set(sentiments))
    
    # Assign an index to each sentiment
    return {s: i for i, s in enumerate(unique)}


def load_ner_sa_data_unbalanced(test_size: float = 0.1) -> Tuple[NER_SA_Dataset, NER_SA_Dataset, torch.Tensor, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Loads and prepares data for multitask training (NER + Sentiment Analysis).
    This version does not balance the data before the train/test split.

    Args:
        test_size: Proportion of the dataset to include in the validation split (default is 0.1).

    Returns:
        A tuple containing:
            - train_dataset: The training dataset for multitask learning.
            - val_dataset: The validation dataset for multitask learning.
            - word_embeddings: Pre-trained word embeddings for the words in the vocabulary.
            - word2idx: A dictionary mapping words to indices.
            - tag2idx: A dictionary mapping NER tags to indices.
            - sentiment2idx: A dictionary mapping sentiment labels to indices.
    """
    
    # 1. Read CSV and parse columns as lists
    df = pd.read_csv(DATASET)
    df["tokens"] = df["tokens"].apply(eval)
    df["ner_tags"] = df["ner_tags"].apply(eval)
    df["sentiment"] = df["sentiment"].astype(str)

    texts = df["tokens"].tolist()
    tags = df["ner_tags"].tolist()
    sentiments = df["sentiment"].tolist()

    # 2. Split data into training and validation sets
    texts_train, texts_val, tags_train, tags_val, sents_train, sents_val = train_test_split(
        texts, tags, sentiments, test_size=test_size, random_state=42
    )

    # 3. Build vocabularies for words, NER tags, and sentiments
    word2idx = build_word2idx(texts_train)
    tag2idx = build_tag2idx(tags_train)
    sentiment2idx = build_sentiment2idx(sents_train)

    # 4. Train Word2Vec model to obtain word embeddings
    w2v_model = Word2Vec(sentences=texts_train, vector_size=300, window=5, min_count=1, workers=4)
    word_embeddings = torch.zeros(len(word2idx), 300)
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            word_embeddings[idx] = torch.tensor(w2v_model.wv[word])
        else:
            word_embeddings[idx] = torch.randn(300)

    # 5. Create multitask datasets
    train_dataset = NER_SA_Dataset(texts_train, tags_train, sents_train, tag2idx, sentiment2idx)
    val_dataset = NER_SA_Dataset(texts_val, tags_val, sents_val, tag2idx, sentiment2idx)

    return train_dataset, val_dataset, word_embeddings, word2idx, tag2idx, sentiment2idx


def load_ner_sa_data(test_size: float = 0.1) -> Tuple[NER_SA_Dataset, NER_SA_Dataset, torch.Tensor, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Loads and prepares data for multitask training (NER + Sentiment Analysis), 
    removing 2000 positive examples before the train/test split.

    Args:
        test_size: Proportion of the dataset to include in the validation split.

    Returns:
        A tuple containing:
            - train_dataset: The training dataset for multitask learning.
            - val_dataset: The validation dataset for multitask learning.
            - word_embeddings: Pre-trained word embeddings for the words in the vocabulary.
            - word2idx: A dictionary mapping words to indices.
            - tag2idx: A dictionary mapping NER tags to indices.
            - sentiment2idx: A dictionary mapping sentiment labels to indices.
    """
    
    # 1. Read CSV and parse columns as lists
    df = pd.read_csv(DATASET)
    df["tokens"] = df["tokens"].apply(eval)
    df["ner_tags"] = df["ner_tags"].apply(eval)
    df["sentiment"] = df["sentiment"].astype(str)

    # 1.1. Remove 2000 "positive" examples before splitting
    df_positive = df[df["sentiment"] == "positive"]
    df_negative = df[df["sentiment"] != "positive"]

    if len(df_positive) > 2000:
        df_positive = df_positive.sample(len(df_positive) - 2000, random_state=42)

    df = pd.concat([df_positive, df_negative]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 2. Separate columns into lists
    texts = df["tokens"].tolist()
    tags = df["ner_tags"].tolist()
    sentiments = df["sentiment"].tolist()

    # 3. Split data into training and validation sets
    texts_train, texts_val, tags_train, tags_val, sents_train, sents_val = train_test_split(
        texts, tags, sentiments, test_size=test_size, random_state=42
    )

    # 4. Build vocabularies for words, NER tags, and sentiments
    word2idx = build_word2idx(texts_train)
    tag2idx = build_tag2idx(tags_train)
    sentiment2idx = build_sentiment2idx(sents_train)

    # 5. Train Word2Vec model to obtain word embeddings
    w2v_model = Word2Vec(sentences=texts_train, vector_size=300, window=5, min_count=1, workers=4)
    word_embeddings = torch.zeros(len(word2idx), 300)
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            word_embeddings[idx] = torch.tensor(w2v_model.wv[word])
        else:
            word_embeddings[idx] = torch.randn(300)

    # 6. Create multitask datasets
    train_dataset = NER_SA_Dataset(texts_train, tags_train, sents_train, tag2idx, sentiment2idx)
    val_dataset = NER_SA_Dataset(texts_val, tags_val, sents_val, tag2idx, sentiment2idx)

    return train_dataset, val_dataset, word_embeddings, word2idx, tag2idx, sentiment2idx

def ner_sa_collate_fn(batch: List[Tuple[List[str], torch.Tensor, torch.Tensor]]) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
    """
    Collate function for batching in NER + Sentiment Analysis
    
    Args:
        batch: A batch of samples where each sample is a tuple (tokens, ner_targets, sa_targets).

    Returns:
        A tuple containing:
            - tokens: A list of token lists.
            - ner_targets_padded: A padded tensor to the maximum length of the batch for NER targets.
            - sa_targets_tensor: A tensor for sentiment analysis targets.
    """
    
    tokens, ner_targets, sa_targets = zip(*batch)

    # Pad NER sequences to the maximum sequence length in the batch (LongTensor)
    ner_targets_padded = pad_sequence(ner_targets, batch_first=True, padding_value=-100)

    # Convert sentiment analysis targets to a tensor
    sa_targets_tensor = torch.stack(sa_targets)

    return list(tokens), ner_targets_padded, sa_targets_tensor



if __name__ == "__main__":
    
    # Load the data for multitask learning (NER + Sentiment Analysis)
    train_dataset, val_dataset, word_embeddings, word2idx, tag2idx, sentiment2idx = load_ner_sa_data()

    # Create an inverse dictionary for sentiment labels (idx -> sentiment)
    idx2sentiment = {v: k for k, v in sentiment2idx.items()}

    # Initialize a Counter to count sentiment occurrences
    from collections import Counter
    sentiment_counts = Counter()

    # Iterate through the training dataset to count sentiment labels
    for _, _, sentiment_tensor in train_dataset:
        sentiment_idx = sentiment_tensor.item()  # Get the sentiment index
        sentiment_label = idx2sentiment[sentiment_idx]  # Convert index to label
        sentiment_counts[sentiment_label] += 1  # Increment the count for this sentiment

    print("\n📊 Distribución de clases en el conjunto de entrenamiento:")
    for label, count in sentiment_counts.items():
        print(f"  {label.capitalize()}: {count} ejemplos")
