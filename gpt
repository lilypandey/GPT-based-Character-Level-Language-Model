import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Hyperparameters
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
BATCH_SIZE = 64
BLOCK_SIZE = 128  # Context window
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask):
        N, query_len, embed_size = query.shape
        _, key_len, _ = key.shape

        query = query.view(N, query_len, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(N, key_len, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nhqk", query, key) / np.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(energy, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", attention, value).reshape(N, query_len, self.embed_size)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        return self.norm2(x + self.dropout(forward))

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout, forward_expansion):
        super(GPTLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(DEVICE)
        x = self.dropout(self.embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

# Example training loop
VOCAB_SIZE = 100  # Example vocab size
model = GPTLanguageModel(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT, forward_expansion=4).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training function
def train(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, None)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Inference function
def generate_text(model, start_text, length=100):
    model.eval()
    input_ids = torch.tensor([ord(c) for c in start_text]).unsqueeze(0).to(DEVICE)
    generated = list(start_text)
    for _ in range(length):
        with torch.no_grad():
            logits = model(input_ids, None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            generated.append(chr(next_char))
            input_ids = torch.cat((input_ids, torch.tensor([[next_char]]).to(DEVICE)), dim=1)
    return "".join(generated)

criterion = nn.CrossEntropyLoss()
print("Model initialized and ready for training!")
