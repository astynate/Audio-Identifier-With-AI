from __dataset__.dataset_processor import DatasetProcessor
from __classes__.audio_processor import AudioProcessor
from __neural_network__.audio_separator_trainer import AudioSeparatorTrainer
from torch.utils.data import Dataset, DataLoader
from __neural_network__.audio_separator import AudioSeparator

INPUT_FILEPATH = '__dataset__/samples/sample_0/mix.wav'
OUTPUT_FILEPATH = '__neural_network__/output/result.wav'

def use_neural_network(input_filepath: str, output_filepath: str, neural_network: AudioSeparator) -> None:
    audio_processor = AudioProcessor()
    # Load Audio From "input_filepath"
    waveform_audio = audio_processor.load_audio(input_filepath)
    # Convert To The Spectrogram
    spectrogram_audio = audio_processor.to_spectrogram(waveform_audio)
    # Forward Propagation Of The Neural Network "neural_network"
    neural_network_result = neural_network.forward(spectrogram_audio)
    # Convert Back To Wave Form
    # neural_network_result_as_waveform = audio_processor.to_waveform(neural_network_result)
    # # Save The Result Audio Into "output_filepath"
    # audio_processor.save_audio(neural_network_result_as_waveform, output_filepath)

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    model = AudioSeparator()

    dataset = DatasetProcessor('__dataset__', audio_processor, num_samples=1)
    audio_separator_trainer = AudioSeparatorTrainer(model)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    audio_separator_trainer.train(dataloader, 10)
    model.save("separator.pth")

    use_neural_network(INPUT_FILEPATH, OUTPUT_FILEPATH, model)