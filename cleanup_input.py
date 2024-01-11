import os
import re

def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s,\-]', '', text) # Remove special characters
    return cleaned_text

def remove_special_characters(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            cleaned_lines = [clean_text(line) for line in lines]  # Remove special characters from each line
            cleaned_lines = [line.rstrip() if line.startswith(' ') else line for line in cleaned_lines]  # Remove trailing space if line starts with a space
            cleaned_text = ''.join(cleaned_lines)
            
        new_file_path = file_path.replace('.txt', '_NEW.txt')
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            new_file.write(cleaned_text)
        
        print(f"Special characters and lines starting with numbers removed. New file saved as {new_file_path}")
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def process_files_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                remove_special_characters(file_path)

folder_path = 'source'
process_files_in_folder(folder_path)
