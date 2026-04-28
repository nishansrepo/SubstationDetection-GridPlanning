import os

def save_directory_tree(parent_dir, output_filename):
    """
    Reads and saves a visual tree of all sub-folders and files 
    inside the given parent directory to a text file.
    """
    # Open the target file in write mode with UTF-8 encoding (to support emojis)
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Check if the directory exists
        if not os.path.exists(parent_dir):
            f.write(f"Error: The directory '{parent_dir}' does not exist.\n")
            return

        f.write(f"Directory listing for: {parent_dir}\n\n")

        # Walk through the directory
        for root, dirs, files in os.walk(parent_dir):
            # Calculate the indentation level based on the depth of the folder
            level = root.replace(parent_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            
            # Write the current folder
            folder_name = os.path.basename(root)
            if not folder_name:  # Fallback for the root path like '.'
                folder_name = parent_dir
            f.write(f"{indent}📂 {folder_name}/\n")
            
            # Write the files inside the current folder
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{sub_indent}📄 {file}\n")

# --- Example Usage ---
# Replace '.' with your target folder path (e.g., 'C:/Users/Name/Documents')
target_directory = "." 
output_file = "directory_tree.txt"

save_directory_tree(target_directory, output_file)
print(f"Directory tree successfully saved to {output_file}")