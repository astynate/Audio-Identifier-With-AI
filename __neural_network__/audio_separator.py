import torch
import torch.nn as nn

class AudioSeparator(nn.Module):
    def __init__(self, n_fft=400):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, mix_spec_complex):
        mix_mag = mix_spec_complex.abs()
        mix_phase = mix_spec_complex.angle()

        mask = self.net(mix_mag)

        est_vocal_mag = mix_mag * mask
        est_vocal_complex = torch.polar(est_vocal_mag, mix_phase)

        return est_vocal_complex

    def save(self, output_filepath: str) -> None:
        torch.save(self.state_dict(), output_filepath)

    def load(self, input_filepath) -> None:
        self.load_state_dict(torch.load(input_filepath))
        model.eval()