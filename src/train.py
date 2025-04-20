import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Own libraries
from src.train_functions import train_step, val_step

from src.Mymodels.BigNet import BigNet
from src.Mymodels.SmallNet import SmallNet

from src.data.data_functions import load_ner_sa_data, ner_sa_collate_fn, build_char2idx

from src.utils import set_seed, save_model



# Hyperparameters
EPOCHS: int = 20
BATCH_SIZE: int = 16
LR: float = 1e-3
HIDDEN_DIM: int = 128
WORD_EMB_DIM: int = 300
CHAR_EMB_DIM: int = 25
# model_name: str = f"BigModel-E{EPOCHS}_BS{BATCH_SIZE}_LR{LR}_scheduler_H{HIDDEN_DIM}_WEMB{WORD_EMB_DIM}_CEMB{CHAR_EMB_DIM}_databalanced"
model_name: str = "Best_model"

set_seed(42)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_dataset, val_dataset, word_embeddings, word2idx, tag2idx, sentiment2idx = load_ner_sa_data()

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ner_sa_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ner_sa_collate_fn)

# Char2idx
char2idx = build_char2idx()

# Model initialization
model = BigNet(
    word_vocab_size=len(word2idx),
    word_emb_dim=WORD_EMB_DIM,
    pretrained_word_embeddings=word_embeddings,
    lstm_hidden_dim=HIDDEN_DIM,
    ner_num_classes=len(tag2idx),
    sentiment_num_classes=len(sentiment2idx),
    char_emb_dim=CHAR_EMB_DIM,
    label_map=tag2idx,
    char2idx=char2idx
).to(device)

model2 = SmallNet(
    word_vocab_size=len(word2idx),
    word_emb_dim=WORD_EMB_DIM,
    pretrained_word_embeddings=word_embeddings,
    lstm_hidden_dim=HIDDEN_DIM,
    ner_num_classes=len(tag2idx),
    sentiment_num_classes=len(sentiment2idx),
    char_emb_dim=CHAR_EMB_DIM,
    label_map=tag2idx,
    char2idx=char2idx
).to(device)

# Optimizer and loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Loss functions
loss_fn_ner = torch.nn.CrossEntropyLoss(ignore_index=-100)
loss_fn_sa = torch.nn.CrossEntropyLoss()

# TensorBoard writer
writer = SummaryWriter(f"runs/{model_name}")

# Training loop
for epoch in tqdm(range(EPOCHS), desc="Entrenando por epochs", unit="epoch"):
    print(f"\n🔁 Epoch {epoch + 1}/{EPOCHS}")

    # Training step
    train_acc, train_f1, train_total_loss, train_sa_loss, train_ner_loss = train_step(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn_ner=loss_fn_ner,
        loss_fn_sa=loss_fn_sa,
        word2idx=word2idx,
        epoch=epoch,
        writer=writer,
        device=device
    )
    
    # Validation step
    val_acc, val_f1, val_total_loss, val_sa_loss, val_ner_loss  = val_step(
        model=model,
        val_loader=val_loader,
        loss_fn_ner=loss_fn_ner,
        loss_fn_sa=loss_fn_sa,
        word2idx=word2idx,
        epoch=epoch,
        writer=writer,
        device=device
    )
    
    # Step the scheduler based on validation loss
    scheduler.step(val_total_loss) 
    
    # Print the results for this epoch
    print(f"✅ Train SA Accuracy: {train_acc:.4f}")
    print(f"🧠 Train NER F1 Score: {train_f1:.4f}")
    print(f"📉 Train Total Loss: {train_total_loss:.4f} | SA Loss: {train_sa_loss:.4f} | NER Loss: {train_ner_loss:.4f}")
    print(f"✅ Val   SA Accuracy: {val_acc:.4f}")
    print(f"🧠 Val   NER F1 Score: {val_f1:.4f}")
    print(f"📉 Val   Total Loss: {val_total_loss:.4f} | SA Loss: {val_sa_loss:.4f} | NER Loss: {val_ner_loss:.4f}")



# Save the model
save_model(model, model_name)

