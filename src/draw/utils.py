import os

def ensure_directories_exist(file_path):
    directory_path = os.path.dirname(file_path)
    normalized_path = os.path.normpath(directory_path)
    if not os.path.exists(normalized_path):
        print(f"Creating directory: {normalized_path}")
        os.makedirs(normalized_path)
    else:
        print(f"Directory already exists: {normalized_path}")

if __name__ == '__main__':
    file_path_input = input("Enter the file path you want to ensure directories exist for: ")
    ensure_directories_exist(file_path_input)
