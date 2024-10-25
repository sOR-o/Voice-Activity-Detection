import os
import glob

def merge_time_segments_in_directory(directory_path):
    # Get a list of all .txt files in the directory
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

    for file_path in txt_files:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Initialize variables for start and end times
                first_start_time = None
                last_end_time = None

                for line in lines:
                    # Split the line by tabs and extract start and end times
                    _, start_time, end_time = line.strip().split('\t')

                    # Set the first start time
                    if first_start_time is None:
                        first_start_time = float(start_time)

                    # Continuously update the last end time
                    last_end_time = float(end_time)

            # Write the merged result back to the same file
            if first_start_time is not None and last_end_time is not None:
                result = f"S\t{first_start_time}\t{last_end_time}\n"
                
                with open(file_path, 'w') as file:
                    file.write(result)
                # print(f"Processed file: {file_path}")

        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

# Example usage
directory_path = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/testing/188_trans"  # Provide the path to your directory containing .txt files
merge_time_segments_in_directory(directory_path)
