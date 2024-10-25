import os

def print_txt_files_content(directory_path):
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            print(f"Contents of {filename}:\n")
            # Open and read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(content)
                print("\n" + "="*50 + "\n")  # Separator between files

# Replace 'your_directory_path' with the actual directory containing your .txt files
directory_path = "/Users/saurabh/Documents/projects/Voice-Activity-Detection/data/bh_dataset/188_samples/188_trans"
print_txt_files_content(directory_path)
