import os
import logging
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from speechbrain.inference.VAD import VAD

def setup_logging(log_path):
    """
    Sets up logging configuration to write logs to a specified file.
        Args:
            log_path (str): Path where logs will be stored.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_path)
    logger = logging.getLogger(__name__)
    return logger

def load_processed_files(log_path):
    """
    Loads the list of already processed files from the log.
        Args:
            log_path (str): Path to the log file.
        Returns:
            list: List of already processed audio files.
    """
    processed_files = []
    
    if os.path.exists(log_path):
        with open(log_path, "r") as file:
            for line in file:
                if "Already processed audio file" in line:
                    processed_files.append(line.split(":")[-1].strip())
                    
    return processed_files

def get_input_from_user():
    """
    Prompts the user interactively to provide the directory paths for audio files and output labels.
        Returns:
            Tuple: A tuple containing the paths to the audio directory and the output directory.
    """
    audio_dir = input("audio_dir: ").strip()
    output_dir = input("output_dir: ").strip()
    return audio_dir, output_dir

def get_paths():
    """
    Retrieves the paths for the audio and output directories. Tries to get them from the command-line arguments first.
    If not provided via command-line, it prompts the user interactively.
        Returns:
            Tuple: A tuple containing the paths to the audio directory and the output directory.
    """
    parser = argparse.ArgumentParser(description="Process audio files with SpeechBrain VAD model.")
    parser.add_argument('--audio_dir', type=str, help="Path to the directory containing audio files (.wav)")
    parser.add_argument('--output_dir', type=str, help="Path to the directory where output labels will be saved")
    
    args = parser.parse_args()

    if args.audio_dir and args.output_dir:
        return args.audio_dir, args.output_dir
    else:
        print("Command-line args missing.")
        return get_input_from_user()

def initialize_vad_model():
    """
    Initializes the Voice Activity Detection (VAD) model from SpeechBrain.
        Returns:
            VAD: An instance of the initialized SpeechBrain VAD model.
    """
    vad_model = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir="../models/.cache"
    )
    return vad_model

def get_txt_filename(audio_path):
    """
    Converts the audio (.wav) file path to a corresponding .txt filename.
        Args:
            audio_path (str): Path to the input audio file.
        Returns:
            str: The corresponding .txt filename for the audio file.
    """
    filename = os.path.basename(audio_path)
    return filename.replace(".wav", ".txt")

def save_speech_segments(segments, output_dir, txt_filename):
    """
    Saves speech segments to a text file in the required format (start_time, end_time, "speech").
        Args:
            segments (list): List of speech segments with start and end times.
            output_dir (str): Directory where the output text file will be saved.
            txt_filename (str): Name of the output text file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, txt_filename)
    
    with open(output_path, 'w') as file:
        for segment in segments:
            start_time, end_time = segment['speech']
            file.write(f"{start_time:.6f}\t{end_time:.6f}\tspeech\n")

def extract_speech_segments(boundaries):
    """
    Extracts and rounds speech segment timestamps from the boundary information.
        Args:
            boundaries (list): List of boundary start and end times for speech segments.
        Returns:
            list: List of dictionaries representing speech segments, each containing start and end times.
    """
    speech_segments = []
    for start, end in boundaries:
        segment = {'speech': [round(start.numpy().item(), 6), round(end.numpy().item(), 6)]}
        speech_segments.append(segment)
    return speech_segments

def list_files_with_extension(directory, file_extension):
    """
    Retrieves a list of all file paths with a given extension in the specified directory.
        Args:
            directory (str): Directory to search for files.
            file_extension (str): File extension to filter by (e.g., '.wav').
        Returns:
            list: List of file paths with the specified extension.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def perform_vad_inference(audio_path, vad_model):
    """
    Performs Voice Activity Detection (VAD) inference on the provided audio file and extracts speech segments.
        Args:
            audio_path (str): Path to the audio file for VAD processing.
            vad_model (VAD): Initialized SpeechBrain VAD model.
        Returns:
            list: List of speech segment boundaries (start and end times).
    """
    # Get speech probabilities and apply a threshold
    prob_chunks = vad_model.get_speech_prob_file(audio_path)
    prob_thresholded = vad_model.apply_threshold(prob_chunks).float()
    
    # Extract boundaries and refine them
    boundaries = vad_model.get_boundaries(prob_thresholded)
    boundaries = vad_model.energy_VAD(audio_path, boundaries)
    boundaries = vad_model.merge_close_segments(boundaries, close_th=0.250)
    boundaries = vad_model.remove_short_segments(boundaries, len_th=0.250)
    boundaries = vad_model.double_check_speech_segments(boundaries, audio_path, speech_th=0.5)
    
    return boundaries

def process_audio_file(audio_path, vad_model, output_directory, logger):
    """
    Processes a single audio file, performs VAD, and saves the speech segments to a text file.
        Args:
            audio_path (str): Path to the audio file for processing.
            vad_model (VAD): Initialized SpeechBrain VAD model.
            output_directory (str): Directory where output labels will be saved.
            logger (logging.Logger): Logger instance for logging file processing information.
    """
    try:
        speech_segments_boundaries = perform_vad_inference(audio_path, vad_model)
        speech_segments = extract_speech_segments(speech_segments_boundaries)
        txt_filename = get_txt_filename(audio_path)
        save_speech_segments(speech_segments, output_directory, txt_filename)
        logger.info(f"Already processed audio file: {audio_path}")
    except Exception as exc:
        logger.error(f"Audio file {audio_path} generated an exception: {exc}")

def main():
    this = os.path.abspath(os.path.dirname(__file__))
    log_path = os.path.join(this, "vad.log")
    
    logger = setup_logging(log_path)
    processed_files = load_processed_files(log_path)
    print(f"already processed files: {len(processed_files)}")

    vad_model = initialize_vad_model()
    audio_directory, output_directory = get_paths()
    
    audio_files = list_files_with_extension(audio_directory, '.wav')
    audio_files = [audio_path for audio_path in audio_files if audio_path not in processed_files]
    
    # Process files using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_audio_file, audio_path, vad_model, output_directory, logger): audio_path for audio_path in audio_files}
        
        for future in tqdm(as_completed(futures), total=len(audio_files), desc="Processing audio files"):
            future.result()  # Check : raised exceptions

if __name__ == "__main__":
    main()

# audio: /Users/saurabh/Documents/projects/Voice-Activity-Detection/data/bh_dataset/bh_audios
# output: /Users/saurabh/Documents/projects/Voice-Activity-Detection/large-eval/vad_label
# run: python /Users/saurabh/Documents/projects/Voice-Activity-Detection/evaluation/large-eval/vad.py --audio_dir /Users/saurabh/Documents/projects/Voice-Activity-Detection/data/bh_dataset/bh_audios --output_dir /Users/saurabh/Documents/projects/Voice-Activity-Detection/large-eval/vad_label