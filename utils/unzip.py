import zipfile
import os

def unzip_file(zip_path, extract_to):
    """
    Unzips the file at `zip_path` and extracts its contents into the `extract_to` directory.

    Parameters:
    zip_path (str): Path to the .zip file.
    extract_to (str): Directory where contents should be extracted.
    """

    # Ensure the zip file exists
    if not os.path.exists(zip_path):
        print(f"Error: Zip file '{zip_path}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open and extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted '{zip_path}' to '{extract_to}'.")

# Example usage:
zip_file_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/archive (4).zip'             # Replace with your zip file path
output_directory = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/data/other'    # Replace with your desired output directory

unzip_file(zip_file_path, output_directory)
