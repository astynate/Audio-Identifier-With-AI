import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

def plot_spec(spec, title, path):
    plt.figure(figsize=(8, 4))

    # spec: [B, C, F, T] или [B, F, T]
    if spec.dim() == 4:  # [B, C, F, T]
        spec = spec[0, 0]  # берём первый элемент батча и первый канал
    elif spec.dim() == 3:  # [B, F, T]
        spec = spec[0]     # берём первый элемент батча
    # теперь spec имеет форму [F, T]

    spec_to_show = torch.log1p(spec.detach().cpu().abs())
    plt.imshow(spec_to_show.numpy(), origin='lower', aspect='auto', cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

class AudioSeparatorTrainer:
    def __init__(self, audio_separator):
        self.audio_separator = audio_separator

    # Device "cuda" or "cpu"
    # def train(self, dataloader, epochs, lr=1e-3, device="cpu"):
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(self.audio_separator.parameters(), lr=lr)

    #     self.audio_separator.to(device)

    #     for epoch in range(epochs):
    #         total_loss = 0

    #         for mix_spec, vocal_spec in dataloader:
    #             mix_spec, vocal_spec = mix_spec.to(device), vocal_spec.to(device)
    #             pred = self.audio_separator(mix_spec)

    #             pred_mag = torch.abs(pred.squeeze(1))
    #             target_mag = torch.abs(vocal_spec)

    #             loss = criterion(pred_mag, target_mag)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item()

    #         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def train(self, train_loader, val_loader=None, epochs=10, lr=1e-3, device="cpu",
        log_every_batches=50, out_dir="__neural_network__/runs", grad_clip=None):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.audio_separator.parameters(), lr=lr)
        self.audio_separator.to(device)

        best_val = float("inf")
        patience, wait = 5, 0  # ранняя остановка

        for epoch in range(1, epochs + 1):
            self.audio_separator.train()
            total_loss, batch_count = 0.0, 0

            for b, (mix_spec, vocal_spec) in enumerate(train_loader, start=1):
                mix_spec = mix_spec.to(device)
                vocal_spec = vocal_spec.to(device)

                pred = self.audio_separator(mix_spec)
                # Align shapes
                if pred.dim() == 4 and pred.size(1) == 1:
                    pred = pred.squeeze(1)

                pred_mag = torch.abs(pred)
                target_mag = torch.abs(vocal_spec)

                loss = criterion(pred_mag, target_mag)

                optimizer.zero_grad()
                loss.backward()

                # Optional grad clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.audio_separator.parameters(), grad_clip)

                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if epoch % 5 == 0:
                    base = f"{out_dir}/epoch_{epoch}/batch_{b}"
                    plot_spec(mix_spec, "Input mix (mag, log1p)", base + "_mix.png")
                    plot_spec(vocal_spec, "Target vocal (mag, log1p)", base + "_target.png")
                    plot_spec(pred, "Predicted vocal (mag, log1p)", base + "_pred.png")
                    plot_spec(torch.abs(pred) - torch.abs(vocal_spec), "Error |pred|-|target|", base + "_error.png")

            avg_train = total_loss / max(1, batch_count)
            print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train:.6f}")

            # # Валидация
            # if val_loader is not None:
            #     self.audio_separator.eval()
            #     val_loss, val_batches = 0.0, 0
            #     with torch.no_grad():
            #         for mix_val, vocal_val in val_loader:
            #             mix_val = mix_val.to(device)
            #             vocal_val = vocal_val.to(device)
            #             pred_val = self.audio_separator(mix_val)
            #             if pred_val.dim() == 4 and pred_val.size(1) == 1:
            #                 pred_val = pred_val.squeeze(1)
            #             val_loss += criterion(torch.abs(pred_val), torch.abs(vocal_val)).item()
            #             val_batches += 1

            #     avg_val = val_loss / max(1, val_batches)
            #     print(f"Epoch {epoch}/{epochs} — Val Loss: {avg_val:.6f}")

            #     # Ранняя остановка
            #     if avg_val < best_val:
            #         best_val, wait = avg_val, 0
            #         # Сохраните лучшую модель
            #         os.makedirs(out_dir, exist_ok=True)
            #         torch.save(self.audio_separator.state_dict(), os.path.join(out_dir, "best_model.pt"))
            #     else:
            #         wait += 1
            #         if wait >= patience:
            #             print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val:.6f}")
            #             break