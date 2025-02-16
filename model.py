import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector, attention_weights

class EmotionClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.embed_dropout = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )
        
        self.attention = AttentionLayer(config.HIDDEN_DIM * 2)
        
        self.fc1 = nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2)
        self.fc3 = nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        
        self.layer_norm1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.layer_norm2 = nn.LayerNorm(config.HIDDEN_DIM // 2)
        
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        
    def forward(self, x):
        x = self.embed_dropout(x)
        
        lstm_out, _ = self.lstm(x)
        
        context, _ = self.attention(lstm_out)
        
        out = self.fc1(context)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out