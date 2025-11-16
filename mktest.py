import os
import random
import shutil

ROOT_DIR = "train/"                 # Starting directory
TARGET_DIR = "val/"
INCLUDE_HIDDEN = False              # Whether to include files starting with .
MIN_FILES = 4                       # Minimum number of files threshold

def main():
    # Iterate through each entry in ROOT_DIR
    for root, dirs, _ in os.walk(ROOT_DIR):
        # If not including hidden files, filter them out
        
        # Only interested in the immediate subdirectories of ROOT_DIR
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            files = [
                f for f in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, f)) and (INCLUDE_HIDDEN or not f.startswith('.'))
            ]
            # Count files in the subdirectory
            num_files = len([f for f in os.listdir(subdir_path)
                             if os.path.isfile(os.path.join(subdir_path, f))
                             and (INCLUDE_HIDDEN or not f.startswith('.'))])

            if num_files>MIN_FILES:
                target_subdir = os.path.join(TARGET_DIR, subdir)
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)
                selected_file = random.choice(files)
                src_file_path = os.path.join(subdir_path, selected_file)
                dst_file_path = os.path.join(target_subdir, selected_file)

                # 移动文件
                shutil.move(src_file_path, dst_file_path)
                print(f"Moved '{selected_file}' from '{subdir_path}' to '{target_subdir}'.")
                

if __name__ == "__main__":
    main()
