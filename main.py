from __neural_network__.audio_separator import AudioSeparator
from __classes__.audio_processor import AudioProcessor
import torchaudio 

INPUT_FILEPATH = '__dataset__/instrumentals/instrumental1.wav'
OUTPUT_FILEPATH = '__neural_network__/output/result.wav'

def use_neural_network(input_filepath: str, output_filepath: str, neural_network: AudioSeparator) -> None:
    audio_processor = AudioProcessor()

    # Load Audio From "input_filepath"
    waveform_audio = audio_processor.load_audio(input_filepath)
    print(waveform_audio.shape)

    # Convert To The Spectrogram
    spectrogram_audio = audio_processor.to_spectrogram(waveform_audio)
    print(spectrogram_audio.shape)
    print(spectrogram_audio)
    
    # Forward Propagation Of The Neural Network "neural_network"
    neural_network_result = neural_network.forward(spectrogram_audio).squeeze(0)
    print(neural_network_result.shape)

    # Convert Back To Wave Form
    neural_network_result_as_waveform = audio_processor.to_waveform(neural_network_result)
    print(neural_network_result.shape)

    # Save The Result Audio Into "output_filepath"
    audio_processor.save_audio(neural_network_result_as_waveform, output_filepath)

if __name__ == "__main__":
    separator = AudioSeparator()
    use_neural_network(INPUT_FILEPATH, OUTPUT_FILEPATH, neural_network=separator)