import os

def merge_speech_segments(file_path):
    try:
        # Read the file and process the lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if not lines:
            print(f"The file {file_path} is empty.")
            return

        # Extract the first and last time points
        first_start, last_end = None, None
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] == 'S':
                start, end = float(parts[0]), float(parts[1])
                if first_start is None:
                    first_start = start  # First speech start time
                last_end = end  # Last speech end time

        if first_start is not None and last_end is not None:
            # Merge and write to the file
            with open(file_path, 'w') as file:
                file.write(f"{first_start}\t{last_end}\tS\n")
        else:
            print(f"No valid speech segments found in {file_path}.")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def process_directory(directory_path):
    try:
        # Loop through all files in the directory
        for file_name in os.listdir(directory_path):
            # Only process .txt files
            if file_name.endswith('.txt'):
                file_path = os.path.join(directory_path, file_name)
                merge_speech_segments(file_path)

    except Exception as e:
        print(f"An error occurred while processing the directory: {e}")

# Example usage
directory_path = '/Users/saurabh/Documents/projects/Voice-Activity-Detection/testing/label'  
process_directory(directory_path)
