class DatasetProcessor:
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

        self.inverse = T.InverseSpectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

    # Saving N-number Of Samples Before Learning
    def save_dataset(number: int, directory_path: str) -> None:
        pass

    def mix_vocals_with_instrumental() -> None:
        pass

    def create_dataset() -> None:
        pass