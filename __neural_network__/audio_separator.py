import torch
import torch.nn as nn

class AudioSeparator:
    def __init__(self, n_fft=400):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, spec):
        spec = spec.unsqueeze(1)
        mask = self.net(spec)
        
        return spec * mask

    def save(self, output_filepath: str) -> None:
        torch.save(self.state_dict(), output_filepath)

    def load(self, input_filepath) -> None:
        self.load_state_dict(torch.load(input_filepath))
        model.eval()