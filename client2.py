import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from whisper_transcriber import transcribe_audio_files
from bert_embedder import BertEmbedder
from opacus import PrivacyEngine
import torch.nn as nn
import torch.optim as optim
import flwr as fl

# ------------------------
# Fix tokenizers parallelism warning
# ------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------
# Dataset
# ------------------------

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ------------------------
# Data preparation
# ------------------------

def prepare_dataset(csv_path, audio_dir, client_id, total_clients=2):
    embedder = BertEmbedder()
    df = pd.read_csv(csv_path)
    transcripts = transcribe_audio_files(audio_dir)
    df = df[df["Video_id"].astype(str).isin(transcripts.keys())]
   
    print("Filtered dataframe shape:", df.shape)
    print("Class counts:\n", df["sentiment"].value_counts())
    print("Available transcript keys:\n", list(transcripts.keys()))
    print("Filtered Video_ids:\n", df["Video_id"].tolist())

    texts = df["Video_id"].astype(str).map(transcripts).tolist()
    labels = df["sentiment"].astype("category").cat.codes.tolist()
    features = [embedder.embed_text(text) for text in texts]

    total = len(features)
    chunk = total // total_clients
    start = (client_id - 1) * chunk
    end = start + chunk if client_id < total_clients else total

    X = features[start:end]
    y = labels[start:end]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Load data
# ------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X_train, X_test, y_train, y_test = prepare_dataset(
    "data/Sample_Dataset.csv", 
    "data/audio", 
    client_id=2
)

train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader  = DataLoader(AudioDataset(X_test, y_test), batch_size=32)

# ------------------------
# Model
# ------------------------

class FeedForward(nn.Module):
    def __init__(self, input_size=768, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

model = FeedForward().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# ------------------------
# Add DP
# ------------------------

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,        
    max_grad_norm=1.0
)

# ------------------------
# Training
# ------------------------

def train(epochs=1):
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"DP Epsilon after training round: {epsilon:.2f}")

# ------------------------
# FL client
# ------------------------

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_parameters(self, params):
        keys = list(model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        train(epochs=1)  # reduced to 1 local epoch to control DP epsilon
        return self.get_parameters({}), len(train_loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        model.eval()
        correct = total = 0
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                test_loss += loss.item() * y.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_loss /= total
        acc = correct / total
        return test_loss, total, {"accuracy": acc}

# ------------------------
# Flower client start
# ------------------------

fl.client.start_client(
    server_address="localhost:8080",
    client=FLClient().to_client()
)
