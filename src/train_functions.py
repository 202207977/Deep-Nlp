from seqeval.metrics import f1_score
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List, Dict


def train_step(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn_ner: torch.nn.Module,
    loss_fn_sa: torch.nn.Module,
    epoch: int,
    word2idx: Dict[str, int],
    writer: SummaryWriter,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    
    """
    Performs one training step, including forward pass, loss computation, and backpropagation for both 
    Named Entity Recognition (NER) and Sentiment Analysis (SA) tasks. Also logs training metrics to TensorBoard.

    """
    
    model.train()
    
    # Initialize variables to accumulate losses and metrics
    total_loss, total_ner_loss, total_sa_loss = 0.0, 0.0, 0.0
    correct_sa = 0
    total_sa = 0
    true_ner: List[List[str]] = []
    pred_ner: List[List[str]] = []

    # Iterate through the training data in batches
    for tokens, ner_labels, sentiment_labels in train_loader:
        
        # Convert words to their corresponding indices
        word_indices = [
            torch.tensor([word2idx.get(token.lower(), word2idx["<unk>"]) for token in sent], dtype=torch.long)
            for sent in tokens
        ]
        
        # Pad the word indices to have equal sequence lengths in the batch
        word_input = pad_sequence(word_indices, batch_first=True, padding_value=word2idx["<pad>"]).to(device)

        ner_labels = ner_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        # Forward pass: get the model's output for both tasks
        sentiment_output, ner_output = model(word_input, tokens)

        # NER Loss (sequential)
        ner_output_flat = ner_output.view(-1, ner_output.shape[-1])
        ner_labels_flat = ner_labels.view(-1)
        loss_ner = loss_fn_ner(ner_output_flat, ner_labels_flat)

        # SA Loss
        loss_sa = loss_fn_sa(sentiment_output, sentiment_labels)

        # Sentiment Accuracy
        preds_sa = torch.argmax(sentiment_output, dim=1)
        correct_sa += (preds_sa == sentiment_labels).sum().item()
        total_sa += sentiment_labels.size(0)

        # F1-Score NER
        ner_preds = torch.argmax(ner_output, dim=-1).cpu().tolist()
        ner_true = ner_labels.cpu().tolist()

        # Append the NER predictions and true labels to lists (excluding padding labels)
        for pred_seq, true_seq, _ in zip(ner_preds, ner_true, tokens):
            pred_ner.append([model.idx2label[p] for p, t in zip(pred_seq, true_seq) if t != -100])
            true_ner.append([model.idx2label[t] for t in true_seq if t != -100])

        # Optimization
        # Total loss is the sum of both the NER and SA losses
        loss = loss_ner + loss_sa
        
        # Backpropagate the loss and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the losses for logging
        total_loss += loss.item()
        total_ner_loss += loss_ner.item()
        total_sa_loss += loss_sa.item()

    acc_sa = correct_sa / total_sa
    f1_ner = f1_score(true_ner, pred_ner)

    # Log the metrics to TensorBoard
    writer.add_scalar("train/loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("train/ner_loss", total_ner_loss / len(train_loader), epoch)
    writer.add_scalar("train/sa_loss", total_sa_loss / len(train_loader), epoch)
    writer.add_scalar("train/sa_acc", acc_sa, epoch)
    writer.add_scalar("train/ner_f1", f1_ner, epoch)

    return acc_sa, f1_ner, total_loss / len(train_loader), total_sa_loss / len(train_loader), total_ner_loss / len(train_loader)



@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn_ner: torch.nn.Module,
    loss_fn_sa: torch.nn.Module,
    epoch: int,
    word2idx: Dict[str, int],
    writer: SummaryWriter,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    
    """
    Performs the validation step during training, evaluating both Named Entity Recognition (NER)
    and Sentiment Analysis (SA) tasks.

    """
    
    model.eval()
    
    # Initialize variables to keep track of losses and accuracies
    total_loss, total_ner_loss, total_sa_loss = 0.0, 0.0, 0.0
    correct_sa = 0
    total_sa = 0

    # Initialize lists to store predictions and true labels for NER evaluation
    all_ner_preds: List[List[str]] = []
    all_ner_labels: List[List[str]] = []

    # Iterate through the validation data loader
    for tokens, ner_labels, sentiment_labels in val_loader:
        
        # Convert words to indices based on word2idx, and pad sequences
        word_indices = [
            torch.tensor([word2idx.get(token.lower(), word2idx["<unk>"]) for token in sent], dtype=torch.long)
            for sent in tokens
        ]
        word_input = pad_sequence(word_indices, batch_first=True, padding_value=word2idx["<pad>"]).to(device)

        ner_labels = ner_labels.to(device)
        sentiment_labels = sentiment_labels.to(device)

        # Perform a forward pass through the model
        sentiment_output, ner_output = model(word_input, tokens)

        # NER Loss
        ner_output_flat = ner_output.view(-1, ner_output.shape[-1])
        ner_labels_flat = ner_labels.view(-1)
        loss_ner = loss_fn_ner(ner_output_flat, ner_labels_flat)

        # SA Loss
        loss_sa = loss_fn_sa(sentiment_output, sentiment_labels)

        # SA Accuracy
        preds = torch.argmax(sentiment_output, dim=1)
        correct_sa += (preds == sentiment_labels).sum().item()
        total_sa += sentiment_labels.size(0)

        # F1 for NER
        ner_preds = torch.argmax(ner_output, dim=-1).cpu().tolist()
        ner_true = ner_labels.cpu().tolist()

        for pred_seq, true_seq in zip(ner_preds, ner_true):
            
            # Exclude special tokens (-100 for padding) and map the predicted and true indices to labels
            pred_labels = [model.idx2label[p] for p, t in zip(pred_seq, true_seq) if t != -100]
            true_labels = [model.idx2label[t] for t in true_seq if t != -100]
            all_ner_preds.append(pred_labels)
            all_ner_labels.append(true_labels)

        # Accumulate the losses for logging
        loss = loss_ner + loss_sa
        total_loss += loss.item()
        total_ner_loss += loss_ner.item()
        total_sa_loss += loss_sa.item()
        
    # Calculate the SA accuracy
    acc = correct_sa / total_sa
    
    # Calculate the F1 score for NER
    f1 = f1_score(all_ner_labels, all_ner_preds, average='macro', zero_division=0)

    # Log the metrics to TensorBoard
    writer.add_scalar("val/loss", total_loss / len(val_loader), epoch)
    writer.add_scalar("val/ner_loss", total_ner_loss / len(val_loader), epoch)
    writer.add_scalar("val/sa_loss", total_sa_loss / len(val_loader), epoch)
    writer.add_scalar("val/sa_acc", acc, epoch)
    writer.add_scalar("val/ner_f1", f1, epoch)

    return acc, f1, total_loss / len(val_loader), total_sa_loss / len(val_loader), total_ner_loss / len(val_loader)



@torch.no_grad()
def evaluate(model: torch.nn.Module, test_loader: DataLoader, word2idx,device: torch.device) -> float:
    """
    Evaluates the model on the test dataset and computes the sentiment analysis accuracy.
    """
    
    model.eval()
    
    # Initialize variables to track the correct predictions and the total number of samples
    correct_sa = 0
    total = 0

    # Iterate through the test dataset
    for tokens, _, sentiment_labels in test_loader:
        
        # Convert tokens to their corresponding indices based on the word2idx dictionary
        word_input = torch.tensor(
            [[word2idx.get(token.lower(), word2idx["<unk>"]) for token in sent] for sent in tokens],
            dtype=torch.long
        ).to(device)

        sentiment_labels = sentiment_labels.to(device)

        # Forward pass: Get the model's output for sentiment analysis
        sentiment_output, _ = model(word_input, tokens)
        
        # Predict sentiment labels by taking the argmax of the model's output
        preds = torch.argmax(sentiment_output, dim=1)
        
        # Count the number of correct sentiment predictions
        correct_sa += (preds == sentiment_labels).sum().item()
        
        # Update the total number of sentiment labels processed
        total += sentiment_labels.size(0)

    acc = correct_sa / total
    print(f"Sentiment Accuracy: {acc:.4f}")
    return acc
