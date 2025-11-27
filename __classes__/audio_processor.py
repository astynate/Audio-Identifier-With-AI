import torch
import torchaudio
import torchaudio.transforms as T

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=200):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.spectrogram = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=None
        )

        self.inverse = T.GriffinLim(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

    def load_audio(self, filepath):
        waveform, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    # .abs()
    def to_spectrogram(self, waveform):
        return self.spectrogram(waveform)

    def to_waveform(self, spectrogram):
        return torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

    # def to_waveform(self, spectrogram, chunk_size=5000):
    #     chunks = []
    #     for start in range(0, spectrogram.size(-1), chunk_size):
    #         end = start + chunk_size
    #         chunk = spectrogram[..., start:end]
    #         waveform_chunk = self.inverse(chunk)
    #         chunks.append(waveform_chunk)
    #     return torch.cat(chunks, dim=-1)

    def save_audio(self, waveform, filepath):
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim > 2:
            waveform = waveform.squeeze()

        waveform = waveform.detach().cpu()
        torchaudio.save(filepath, waveform, self.sample_rate)

if __name__ == "__main__":
    audio = AudioProcessor(sample_rate=16000)
    waveform = audio.load_audio("../__dataset__/vocals/vocal1.wav")
    spec = audio.to_spectrogram(waveform)
    audio.save_audio(audio.to_waveform(spec), "../__dataset__/vocal.wav")