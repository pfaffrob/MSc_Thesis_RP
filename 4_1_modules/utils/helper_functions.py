"""
General-purpose utility functions for file management and directory operations.
Provides tools for listing files, creating directories, and managing folder structures.
"""

import os


def list_files(folder_path, file_extension=None, include_subfolders=False):
    """
    List all files in a given folder. Optionally filter by file extension.
    If include_subfolders is True, search all subdirectories; if False, only list files in the top-level folder.
    If the folder does not exist, print a message and return an empty list.

    :param folder_path: Path to the folder
    :param file_extension: Optional file extension to filter by (e.g., '.las')
    :param include_subfolders: Whether to include subfolders in the search
    :return: List of file paths
    """
    if not os.path.exists(folder_path):
        print(f"Directory does not exist: {folder_path}")
        return []

    files = []
    if include_subfolders:
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                if not file_extension or filename.endswith(file_extension):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                if not file_extension or filename.endswith(file_extension):
                    files.append(full_path)

    return files



def print_folder_structure(root_dir, show_files=False, max_depth=None):
    root_dir = os.path.abspath(root_dir)
    for root, dirs, files in os.walk(root_dir):
        # Calculate the depth of the current folder
        relative_path = os.path.relpath(root, root_dir)
        level = 0 if relative_path == '.' else relative_path.count(os.sep)

        # Stop if we've reached the maximum depth
        if max_depth is not None and level >= max_depth:
            # Prevent os.walk from descending further
            dirs[:] = []
            continue

        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")

        if show_files:
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")


def update_folder_structure(template_path):
    """
    Adds missing folders from the template_path to all sibling folders
    in the same parent directory. Only creates empty folders if they don't exist.
    
    Parameters:
        template_path (str): Path to the Z_folder_template directory.
    """
    template_path = os.path.abspath(template_path)
    parent_dir = os.path.dirname(template_path)
    template_structure = []

    # Walk through the template and record relative folder paths
    for root, dirs, files in os.walk(template_path):
        for d in dirs:
            full_path = os.path.join(root, d)
            rel_path = os.path.relpath(full_path, template_path)
            template_structure.append(rel_path)

    # Apply the structure to all sibling folders
    for sibling in os.listdir(parent_dir):
        sibling_path = os.path.join(parent_dir, sibling)
        if sibling_path == template_path or not os.path.isdir(sibling_path):
            continue

        for rel_path in template_structure:
            target_folder = os.path.join(sibling_path, rel_path)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                print(f"Created: {target_folder}")

import os
import shutil

def makedirs(path, delete_if_exists = True, exist_ok = True):
    """
    Deletes the directory at `path` if it exists, then recreates it.
    """
    if delete_if_exists == True:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=exist_ok)
