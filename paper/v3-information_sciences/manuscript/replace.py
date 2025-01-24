import re
import argparse
import os

# Function to replace the specified macro with its content in a file
def replace_macro(file_path, macro_name):
    # Create the pattern to match the macro
    pattern = rf'\\{macro_name}\{{([^}}]*)\}}'
    
    # Open the .tex file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Use regular expression to replace the macro with its content
    updated_content = re.sub(pattern, r'\1', content)
    
    # Save the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)

# Function to process all .tex files in a directory and its subdirectories
def process_directory(directory, macro_name):
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):  # Check if the file is a .tex file
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                replace_macro(file_path, macro_name)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Replace a specified macro with its content in all .tex files in a directory and its subdirectories")
    parser.add_argument('directory', type=str, help="Path to the directory containing .tex files")
    parser.add_argument('macro_name', type=str, help="Name of the macro to replace (e.g., 'revise')")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to process the directory
    process_directory(args.directory, args.macro_name)
    
    print(f"Replacement complete for all .tex files!")

if __name__ == "__main__":
    main()
