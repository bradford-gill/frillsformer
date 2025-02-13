import os
import requests
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import cross_entropy

from src.model import Transformer  # Import the Transformer from model.py

class ShakespeareDataset(Dataset):
    """
    This dataset converts the raw Shakespeare text into samples of fixed-length.
    Each sample is a tuple (src, tgt) where:
      - src: a tensor of token indices of length `seq_len`
      - tgt: a tensor of token indices of length `seq_len`, shifted right by one position.
    """
    def __init__(self, text: str, seq_len: int, char_to_idx: dict):
        self.seq_len = seq_len
        self.data = []
        # Convert text into a list of indices.
        indices = [char_to_idx[c] for c in text]
        # Create non-overlapping chunks of (seq_len+1) tokens.
        for i in range(0, len(indices) - seq_len, seq_len):
            chunk = indices[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                src = torch.tensor(chunk[:-1], dtype=torch.long)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                self.data.append((src, tgt))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def download_shakespeare(file_path: str = "shakespeare_input.txt") -> str:
    """
    Downloads the Tiny Shakespeare dataset if it does not already exist.
    """
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def generate_text(model, prompt: str, char_to_idx, idx_to_char, seq_len: int, device, gen_length: int = 200):
    """
    Generates text using greedy decoding.
    Since our Transformer expects a pair of sequences, we use the prompt as both src and tgt,
    and then autoregressively generate one token at a time.
    """
    model.eval()
    # Convert the prompt into a tensor of indices.
    input_ids = [char_to_idx.get(c, 0) for c in prompt]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = prompt

    for _ in range(gen_length):
        # Ensure the input does not exceed seq_len.
        current_seq = input_tensor if input_tensor.size(1) <= seq_len else input_tensor[:, -seq_len:]
        with torch.no_grad():
            # Provide the same sequence as both source and target.
            output = model(current_seq, current_seq)
        # Greedy decoding: select the token with highest probability from the last time step.
        logits = output[0, -1, :]
        next_token = torch.argmax(logits).item()
        generated += idx_to_char[next_token]
        # Append the new token and continue.
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
    return generated

def main():
    # Download and load the Tiny Shakespeare dataset.
    text = download_shakespeare()

    # Build a character-level vocabulary.
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Hyperparameters for our small transformer.
    seq_len = 64        # Sequence length for each training example.
    embed_dim = 32      # Small embedding dimension.
    num_layers = 2      # Use 2 layers for the Transformer.
    num_heads = 2       # Number of attention heads.
    ff_dim = 64         # Feed-forward network dimension.
    dropout = 0.1
    batch_size = 32
    epochs = 5
    learning_rate = 0.001

    # Create the dataset and dataloader.
    dataset = ShakespeareDataset(text, seq_len, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small instance of the Transformer model.
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop.
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt)  # Output shape: [batch_size, seq_len, vocab_size]
            loss = criterion(outputs.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # Generate sample text after training.
    prompt = "To be, or not to be: "
    generated_text = generate_text(model, prompt, char_to_idx, idx_to_char, seq_len, device, gen_length=200)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()