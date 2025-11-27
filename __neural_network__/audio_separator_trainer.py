import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class AudioSeparatorTrainer:
    def __init__(self, audio_separator):
        self.audio_separator = audio_separator

    # Device "cuda" or "cpu"
    def train(self, dataloader, epochs, lr=1e-3, device="cpu"):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.audio_separator.parameters(), lr=lr)

        self.audio_separator.to(device)

        for epoch in range(epochs):
            total_loss = 0

            for mix_spec, vocal_spec in dataloader:
                mix_spec, vocal_spec = mix_spec.to(device), vocal_spec.to(device)

                pred = self.audio_separator(mix_spec)
                loss = criterion(pred.squeeze(1), vocal_spec)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")