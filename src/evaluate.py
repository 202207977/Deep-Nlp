import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


import torch.nn.functional as F

# Own libraries
from src.Mymodels.AlertGenerator import AlertGenerator
from src.Mymodels.BigNet import BigNet
from src.Mymodels.SmallNet import SmallNet

from src.data.data_functions import load_ner_sa_data
from src.data.data_functions import ner_sa_collate_fn 

from src.utils import set_seed


# Hiperparámetros
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 128
WORD_EMB_DIM = 300
CHAR_EMB_DIM = 25
BOOL_ALERT_GENERATION = True

def evaluate(model: torch.nn.Module, alert_generator, test_loader: DataLoader, word2idx, device: torch.device):
    """
    Evaluates the model on the test dataset, computes Sentiment Analysis (SA) accuracy and NER F1-score,
    and optionally generates alerts based on the predictions.
    """
    
    model.eval()
    correct_sa = 0
    total_sa = 0
    pred_ner = []
    true_ner = []
    
    with torch.no_grad():
        for tokens, ner_labels, sentiment_labels in tqdm(test_loader, desc="Evaluando"):
            
            # Convert words to their corresponding indices in the vocabulary
            word_indices = [
                torch.tensor([word2idx.get(token.lower(), word2idx["<unk>"]) for token in sent], dtype=torch.long)
                for sent in tokens
            ]
            word_input = pad_sequence(word_indices, batch_first=True, padding_value=word2idx["<pad>"]).to(device)

            sentiment_labels = sentiment_labels.to(device)
            ner_labels = ner_labels.to(device)

            # Forward pass: Get the model's output for sentiment and NER tasks
            sentiment_output, ner_output = model(word_input, tokens)
            
            
            # === SA Accuracy ===
            preds_sa = torch.argmax(sentiment_output, dim=1)
            correct_sa += (preds_sa == sentiment_labels).sum().item()
            total_sa += sentiment_labels.size(0)

            # === NER F1 ===
            ner_preds = torch.argmax(ner_output, dim=-1).cpu().tolist()
            ner_true = ner_labels.cpu().tolist()

            for pred_seq, true_seq in zip(ner_preds, ner_true):
                pred_ner.append([model.idx2label[p] for p, t in zip(pred_seq, true_seq) if t != -100])
                true_ner.append([model.idx2label[t] for t in true_seq if t != -100])
            
            # === Alert Generation (if enabled) ===
            if alert_generator is not None:
                probs = F.softmax(sentiment_output, dim=1)  # Convert logits to probabilities
                confidences = torch.max(probs, dim=1).values  # Confidence for each sentiment prediction
                alerts = alert_generator.generate_alerts(tokens, ner_preds, preds_sa, confidences)  # Generate alerts
                for a in alerts:
                    print(a)
    
    # Calculate Sentiment Analysis accuracy  
    acc_sa = correct_sa / total_sa
    
    # Calculate NER F1 score using the seqeval library
    f1_ner = f1_score(true_ner, pred_ner)
    

    print(f"🎯 Test Sentiment Accuracy: {acc_sa:.4f}")
    print(f"🧠 Test NER F1-score: {f1_ner:.4f}")

    return acc_sa, f1_ner

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (note: test dataset here is used as the validation set)
    _, test_dataset, word_embeddings, word2idx, tag2idx, sentiment2idx = load_ner_sa_data()

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=ner_sa_collate_fn)


    # Initialize trained models
    model = BigNet(
        word_vocab_size=len(word2idx),
        word_emb_dim=300,
        pretrained_word_embeddings=word_embeddings,
        lstm_hidden_dim=128,
        ner_num_classes=len(tag2idx),
        sentiment_num_classes=len(sentiment2idx),
        char_emb_dim=25,
        label_map=tag2idx,
        char2idx=None  
        )
        
    model2 = SmallNet(
        word_vocab_size=len(word2idx),
        word_emb_dim=300,
        pretrained_word_embeddings=word_embeddings,
        lstm_hidden_dim=128,
        ner_num_classes=len(tag2idx),
        sentiment_num_classes=len(sentiment2idx),
        char_emb_dim=25,
        label_map=tag2idx,
        char2idx=None
    ).to(device)
    
    # Initialize the alert generator if enabled
    alert_generator = None
    if BOOL_ALERT_GENERATION:
        alert_generator = AlertGenerator(tag2idx)

    # Load the pre-trained model weights
    name: str = "Best_model"
    # name = 'models/BigModel-E20_BS16_LR0.001_scheduler_H128_WEMB300_CEMB25_databalanced.pt'
    model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))  # Esto lo he añadido: map_location=torch.device('cpu')
    model.to(device)

    # Evaluate the model's performance on the test set
    evaluate(model, alert_generator, test_loader, word2idx, device)
