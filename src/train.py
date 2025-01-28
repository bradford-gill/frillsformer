import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import cross_entropy

# Define a small Transformer model, Use our model in next PR
class DummyTransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, ff_dim, max_seq_length, num_classes):
        super(DummyTransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, src, tgt):
        # Embed and add positional encoding
        src = self.embedding(src) + self.positional_encoding[:src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :]
        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1))
        return self.fc(output.transpose(0, 1))

# Toy dataset
class ToyDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.data = [
            (torch.randint(0, vocab_size, (seq_length,)), torch.randint(0, vocab_size, (seq_length,)))
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Hyperparameters
input_dim = 50
embed_dim = 32
n_heads = 2
num_encoder_layers = 2
num_decoder_layers = 2
ff_dim = 64
max_seq_length = 20
num_classes = 50
batch_size = 16
epochs = 10
learning_rate = 0.001

# Data
train_dataset = ToyDataset(num_samples=1000, seq_length=max_seq_length, vocab_size=input_dim)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DummyTransformerModel(input_dim, embed_dim, n_heads, num_encoder_layers, num_decoder_layers, ff_dim, max_seq_length, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        # Shift target for decoder (typical seq2seq structure)
        loss = cross_entropy(output.view(-1, num_classes), tgt.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({"Loss": loss.item()})

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

print("Training complete")
