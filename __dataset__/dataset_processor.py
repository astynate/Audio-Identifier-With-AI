import os
import random

class DatasetProcessor:
    def __init__(
        self, 
        dataset_directorypath: str, 
        audio_processor, 
        is_save_dataset: bool = True, 
        sample_rate: int = 10, 
        num_samples: int = 5, 
        duration: int = 10) -> None:

        self.dataset_directorypath = dataset_directorypath
        self.is_save_dataset = True
        self.sample_rate = 16000
        self.audio_processor = audio_processor
        self.data = self.mix_vocals_with_instrumental(num_samples=num_samples, duration=duration)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spec_mix, spec_vocal = self.data[idx]
        return spec_mix, spec_vocal

    def mix_vocals_with_instrumental(self, num_samples: int = 5, duration: int = 10):
        instr_dir = os.path.join(self.dataset_directorypath, "instrumentals")
        vocal_dir = os.path.join(self.dataset_directorypath, "vocals")

        instr_files = [os.path.join(instr_dir, f) for f in os.listdir(instr_dir) if f.endswith(".wav")]
        vocal_files = [os.path.join(vocal_dir, f) for f in os.listdir(vocal_dir) if f.endswith(".wav")]

        results = []
        sample_length = self.sample_rate * duration

        for i in range(num_samples):
            instr_file = random.choice(instr_files)
            vocal_file = random.choice(vocal_files)

            instr_wave = self.audio_processor.load_audio(instr_file)
            vocal_wave = self.audio_processor.load_audio(vocal_file)

            instr_wave = instr_wave[:, :sample_length]
            vocal_wave = vocal_wave[:, :sample_length]

            if instr_wave.size(1) < sample_length:
                instr_wave = torch.nn.functional.pad(instr_wave, (0, sample_length - instr_wave.size(1)))
            if vocal_wave.size(1) < sample_length:
                vocal_wave = torch.nn.functional.pad(vocal_wave, (0, sample_length - vocal_wave.size(1)))

            mix_wave = instr_wave + vocal_wave

            spec_mix = self.audio_processor.to_spectrogram(mix_wave)
            spec_vocal = self.audio_processor.to_spectrogram(vocal_wave)

            results.append((spec_mix, spec_vocal))

            if self.is_save_dataset is True and i % 7 == 0:
                sample_dir = os.path.join(self.dataset_directorypath, "samples", f"sample_{i}")
                os.makedirs(sample_dir, exist_ok=True)

                self.audio_processor.save_audio(vocal_wave, os.path.join(sample_dir, "voice.wav"))
                self.audio_processor.save_audio(mix_wave, os.path.join(sample_dir, "mix.wav"))

        return results