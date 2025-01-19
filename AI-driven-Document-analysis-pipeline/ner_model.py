#src/ner_model.py

import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import LayerNorm

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % self.num_heads == 0, "Hidden dimension should be divisible by the number of heads."
        
        self.head_dim = hidden_dim // self.num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Compute output of attention mechanism
        attention_output = torch.matmul(attention_weights, values)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.out(attention_output)
        
        return output, attention_weights

class BiLSTM_CRF_Attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, embedding_matrix=None, dropout=0.5, num_heads=4):
        super(BiLSTM_CRF_Attn, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) if embedding_matrix is not None else nn.Embedding(vocab_size, embed_dim)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Multi-Head Attention layer
        self.attn_layer = MultiHeadAttentionLayer(hidden_dim, num_heads)
        
        # Layer normalization for LSTM and attention output
        self.layer_norm = LayerNorm(hidden_dim * 2)

        # Fully connected layer for CRF emissions
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding lookup
        embeddings = self.embedding(x)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embeddings)
        
        # Multi-Head Attention mechanism
        attn_out, attn_weights = self.attn_layer(lstm_out)
        
        # Layer Normalization on LSTM and Attention outputs
        lstm_out = self.layer_norm(lstm_out)
        attn_out = self.layer_norm(attn_out)

        # Apply dropout for regularization
        lstm_out = self.dropout(lstm_out)
        attn_out = self.dropout(attn_out)

        # Compute emissions for CRF
        emissions = self.fc(lstm_out)
        
        return emissions, attn_out, attn_weights

    def forward_crf(self, x, tags):
        emissions, _, _ = self.forward(x)
        return -self.crf(emissions, tags)  # Negative log-likelihood for CRF loss

    def predict(self, x):
        emissions, _, _ = self.forward(x)
        return self.crf.decode(emissions)  # Return best tag sequence

# Optimizer (AdamW) for regularization and better convergence
def get_optimizer(model, lr=1e-5, weight_decay=1e-4):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Saving model with enhanced functionality
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Example training function
def train_model(model, train_loader, valid_loader, epochs=10, lr=1e-5):
    optimizer = get_optimizer(model, lr=lr)
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            x, tags = batch  # Assume batch is (input_data, target_tags)
            
            # Compute the loss and backpropagate
            loss = model.forward_crf(x, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation step
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                x, tags = batch
                loss = model.forward_crf(x, tags)
                valid_loss += loss.item()

        # Checkpoint saving (only if validation loss improves)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, "best_model.pth")

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss/len(train_loader)}, Validation Loss: {valid_loss/len(valid_loader)}")

# Example of prediction after training
def predict_tags(model, x):
    model.eval()
    with torch.no_grad():
        return model.predict(x)
