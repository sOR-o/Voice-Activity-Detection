import os
import shutil

def copy_matching_txt_files(wav_dir, txt_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .wav filenames (without extension) from the wav directory
    wav_files = {os.path.splitext(file)[0] for file in os.listdir(wav_dir) if file.endswith('.wav')}
    
    # Get all .txt files from the txt directory
    txt_files = os.listdir(txt_dir)
    
    # Iterate over txt files and copy those that match the .wav filenames
    for txt_file in txt_files:
        if txt_file.endswith('.txt'):
            txt_name = os.path.splitext(txt_file)[0]
            if txt_name in wav_files:
                source_path = os.path.join(txt_dir, txt_file)
                destination_path = os.path.join(output_dir, txt_file)
                shutil.copy(source_path, destination_path)

# Example usage
wav_directory = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/data/bh_dataset/188_samples/188_audio"
txt_directory = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/data/bh_dataset/label"
output_directory = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/testing/test"

copy_matching_txt_files(wav_directory, txt_directory, output_directory)
