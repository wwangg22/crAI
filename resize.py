from PIL import Image
import os

def resize_cover_center_crop(
    input_path, 
    output_path, 
    target_width=1000, 
    target_height=1300
):
    """
    Resizes (cover) and center-crops an image to exactly 1000x1300 px.
    """
    with Image.open(input_path) as im:
        orig_width, orig_height = im.size

        # 1) Compute scale factor (cover)
        #    We want the scaled image to fill at least 1000x1300
        #    so we take the max ratio, not min.
        scale = max(target_width / orig_width, target_height / orig_height)

        # 2) Compute new scaled dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # 3) Resize the image with ANTIALIAS (or LANCZOS in newer Pillow)
        resized = im.resize((new_width, new_height), Image.LANCZOS)

        # 4) Center-crop to 1000x1300 if needed
        #    We find the center and crop out exactly 1000x1300
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        final = resized.crop((left, top, right, bottom))

        # 5) Save to output
        final.save(output_path, quality=95)
        print(f"Saved cropped image: {output_path}")

if __name__ == "__main__":
    # Example usage
    folder = r"C:\Users\niudb\OneDrive\Desktop\cracked_royale\computer_vision\arenas"
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            in_path = os.path.join(folder, filename)
            out_path = os.path.join(folder, f"resized_{filename}")
            resize_cover_center_crop(in_path, out_path, 1000, 1300)
