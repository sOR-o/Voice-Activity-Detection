import os

def count_txt_files(directory):
    return len([file for file in os.listdir(directory) if file.endswith('.txt')])

# Example usage
directory_path = '/Users/saurabh/Documents/projects/Voice-Activity-Detection/comparison/50_trans_for_FE'
txt_file_count = count_txt_files(directory_path)
print(f'Number of .txt files: {txt_file_count}')
directory_path = '/Users/saurabh/Documents/projects/Voice-Activity-Detection/comparison/silero'
txt_file_count = count_txt_files(directory_path)
print(f'Number of .txt files: {txt_file_count}')
directory_path = '/Users/saurabh/Documents/projects/Voice-Activity-Detection/comparison/speechbrain'
txt_file_count = count_txt_files(directory_path)
print(f'Number of .txt files: {txt_file_count}')



