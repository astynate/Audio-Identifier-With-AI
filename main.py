from __neural_network__.audio_separator import AudioSeparator
from __classes__.audio_processor import AudioProcessor

INPUT_FILEPATH = '__dataset__/instrumentals/instrumental1.wav'
OUTPUT_FILEPATH = '__neural_network__/output/result.wav'

def use_neural_network(input_filepath: str, output_filepath: str, neural_network: AudioSeparator) -> None:
    # Load Audio From "input_filepath"
    waveform_audio = audio_processor.load_audio(input_filepath)
    # Convert To The Spectrogram
    spectrogram_audio = audio_processor.to_spectrogram(waveform_audio)
    # Forward Propagation Of The Neural Network "neural_network"
    neural_network_result = separator.forward(spectrogram_audio)
    # Convert Back To Wave Form
    neural_network_result_as_waveform = audio_processor.to_waveform(neural_network_result)
    # Save The Result Audio Into "output_filepath"
    audio_processor.save_audio(neural_network_result_as_waveform, output_filepath)

def use_audio_identifier(input_filepath: str, output_filepath: str) -> None:
    separator = AudioSeparator()
    audio_processor = AudioProcessor()

if __name__ == "__main__":
    use_audio_identifier(INPUT_FILEPATH, OUTPUT_FILEPATH)