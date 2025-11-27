import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSeparator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32+32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16+16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, mix_spec_complex):
        mix_mag = mix_spec_complex.abs()
        mix_phase = mix_spec_complex.angle()

        e1 = self.enc1(mix_mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = self._pad_to_match(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_to_match(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask = self.out_act(self.out_conv(d1))
        mask = F.interpolate(mask, size=mix_mag.shape[2:], mode="bilinear", align_corners=False)

        est_vocal_mag = mix_mag * mask
        est_vocal_complex = torch.polar(est_vocal_mag, mix_phase)

        return est_vocal_complex

    def _pad_to_match(self, x, ref):
        """Pad tensor x so that its H,W match ref's H,W."""
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        return x

    def save(self, output_filepath: str) -> None:
        torch.save(self.state_dict(), output_filepath)

    def load(self, input_filepath) -> None:
        self.load_state_dict(torch.load(input_filepath))
        self.eval()