import os
import glob

st = "building_basic_cannon_sprite_*.png"

def rename_images(folder_path, start_index=2):
    """
    Renames matching PNG files in 'folder_path' by enumerating from 'start_index'.
    For example, if start_index=2, then the first file is (2*10) -> 20.png,
    the second file is (3*10) -> 30.png, etc.
    """
    file_list = sorted(glob.glob(os.path.join(folder_path, st)))

    # Enumerate from start_index
    for i, old_path in enumerate(file_list, start=start_index):
        new_filename = f"{(i-start_index) * 10}.png"
        new_path = os.path.join(folder_path, new_filename)
        
        print(f"Renaming: {old_path} -> {new_path}")
        os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = r"C:\Users\niudb\OneDrive\Desktop\cracked_royale\computer_vision\troops\cannon\friendly"
    rename_images(folder_path, start_index=20)  # Start enumeration at 2
