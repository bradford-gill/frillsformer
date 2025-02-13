# train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from model import Transformer

# Hyperparameters
SRC_VOCAB_SIZE = 10000  # Example values
TGT_VOCAB_SIZE = 10000
EMBED_DIM = 512
MAX_SEQ_LEN = 100
NUM_LAYERS = 6
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 10

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    EMBED_DIM,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    NUM_HEADS,
    FF_DIM,
    DROPOUT
).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens (idx=0)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Example Dataset (replace with your data)
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src = src_data
        self.tgt = tgt_data
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

# Dummy data (replace with actual data loading)
src_data = [torch.randint(1, SRC_VOCAB_SIZE, (10,)) for _ in range(100)]
tgt_data = [torch.randint(1, TGT_VOCAB_SIZE, (12,)) for _ in range(100)]
dataset = TranslationDataset(src_data, tgt_data)

def collate_fn(batch):
    src, tgt = zip(*batch)
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    
    # Generate masks
    src_mask = (src_padded != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, SRC_LEN)
    
    # Decoder self-attention mask (no peeking into future)
    tgt_len = tgt_padded.size(1)
    look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    tgt_padding_mask = (tgt_padded != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_padding_mask & look_ahead_mask.to(device)
    
    return src_padded.to(device), tgt_padded.to(device), src_mask.to(device), tgt_mask.to(device)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for src, tgt, src_mask, tgt_mask in train_loader:
        optimizer.zero_grad()
        
        # Prepare decoder input and output (teacher forcing)
        tgt_input = tgt[:, :-1]  # Exclude last token
        tgt_output = tgt[:, 1:]  # Exclude first token
        
        # Forward pass
        logits = model(
            src=src,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask[:, :, :-1, :-1],  # Adjust mask for shifted tgt_input
            memory_mask=src_mask  # Cross-attention mask (optional)
        )
        
        # Compute loss
        loss = criterion(logits.reshape(-1, TGT_VOCAB_SIZE), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")