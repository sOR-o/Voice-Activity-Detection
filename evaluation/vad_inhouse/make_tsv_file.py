import os

def extract_wav_filenames(directory_path, output_file="/Users/saurabh/Documents/projects/Voice-Activity-Detection/evaluation/vad_inhouse/vad_vaniData/vani_vad_tsv_file.tsv"):
    with open(output_file, 'w') as file:
        for root, _, files in os.walk(directory_path):
            for filename in files:
                if filename.endswith(".wav"):
                    full_file_path = os.path.join(root, filename)
                    file_without_extension = os.path.splitext(filename)[0]
                    file.write(f"{file_without_extension}\t{full_file_path}\n")



directory = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/data/vani_dataset/audios"
extract_wav_filenames(directory)

# run: python /Users/saurabh/Documents/projects/Voice-Activity-Detection/evaluation/vad_inhouse/make_tsv_file.py

