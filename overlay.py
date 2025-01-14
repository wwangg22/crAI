import os
import glob
from PIL import Image

def composite_overlays_with_base(folder_path, base_name="base.png"):
    base_path = os.path.join(folder_path, base_name)
    base_img = Image.open(base_path).convert("RGBA")  # Ensure RGBA

    # Collect overlay files, e.g. 0.png, 10.png, 20.png, ...
    # Exclude base.png so we don't overwrite our base image
    overlay_files = sorted(
        f for f in glob.glob(os.path.join(folder_path, "*.png"))
        if os.path.basename(f) != base_name
    )

    for overlay_path in overlay_files:
        overlay = Image.open(overlay_path).convert("RGBA")

        # Alpha composite: overlay on top of base
        composed = Image.alpha_composite(base_img, overlay)

        # Overwrite the same file (in-place)
        composed.save(overlay_path)
        print(f"Overwrote: {overlay_path}")

if __name__ == "__main__":
    folder = r"C:\Users\niudb\OneDrive\Desktop\cracked_royale\computer_vision\troops\cannon\friendly"
    composite_overlays_with_base(folder)
