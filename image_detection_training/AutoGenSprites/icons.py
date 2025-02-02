import os
import glob
import random
import time

from PIL import Image, ImageEnhance

def apply_color_tint(image, tint_color, intensity=0.6):
    """
    Apply a color tint to an image while preserving the alpha channel.

    :param image: PIL.Image in RGBA mode.
    :param tint_color: Tuple of (R, G, B) values for the tint.
    :param intensity: Float between 0 and 1 indicating the intensity of the tint.
    :return: PIL.Image with the tint applied.
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Split the image into its respective channels
    r, g, b, a = image.split()
    
    # Create a solid color image for the tint
    tint_layer = Image.new('RGBA', image.size, tint_color + (0,))
    
    # Blend the original image with the tint layer
    tinted_rgb = Image.blend(image.convert('RGB'), Image.new('RGB', image.size, tint_color), intensity)
    
    # Reattach the original alpha channel
    tinted_image = Image.merge('RGBA', (*tinted_rgb.split(), a))
    
    return tinted_image

def overlay_icon(base_img, icon_img, x, y):
    """
    Overlays the icon image onto the base image at position (x, y).

    :param base_img: PIL.Image in RGBA mode (background).
    :param icon_img: PIL.Image in RGBA mode (icon to overlay).
    :param x: X-coordinate for the top-left corner.
    :param y: Y-coordinate for the top-left corner.
    :return: Combined PIL.Image.
    """
    overlay = Image.new('RGBA', base_img.size, (0,0,0,0))
    overlay.paste(icon_img, (x, y), icon_img)
    combined = Image.alpha_composite(base_img, overlay)
    return combined

def check_overlap(new_box, dead_zones):
    """
    Checks if the new bounding box overlaps with any existing dead zones.

    :param new_box: Tuple (x1, y1, x2, y2) for the new box.
    :param dead_zones: List of tuples [(x1, y1, x2, y2), ...].
    :return: True if overlaps, False otherwise.
    """
    nx1, ny1, nx2, ny2 = new_box
    for dz in dead_zones:
        dx1, dy1, dx2, dy2 = dz
        # Check for overlap
        if (nx1 < dx2 and nx2 > dx1 and ny1 < dy2 and ny2 > dy1):
            return True
    return False

def extract_class_id(icon_path):
    """
    Extracts the class_id from the icon's filename.
    Assumes the filename is in the format [class_id].png.

    :param icon_path: Path to the icon image.
    :return: Integer class_id or None if extraction fails.
    """
    filename = os.path.basename(icon_path)
    name, ext = os.path.splitext(filename)
    try:
        class_id = int(name)
        return class_id
    except ValueError:
        print(f"Warning: Could not extract class_id from '{filename}'. Skipping this icon.")
        return None

def place_icons(
    base_img,
    icons_dir="./icons",
    num_icons=5,
    max_tries=50,
    dead_zones=None,
    min_scale=0.1,
    max_scale=0.5,
    tint_probability=0.8,
    tint_intensity=0.3,
    rotation_degree=0
):
    """
    Places a specified number of icons onto the base image without overlapping.

    :param base_img: PIL.Image in RGBA mode.
    :param icons_dir: Directory containing icon PNGs.
    :param num_icons: Number of icons to place.
    :param max_tries: Maximum attempts to place each icon.
    :param dead_zones: List to track occupied regions.
    :param min_scale: Minimum scaling factor.
    :param max_scale: Maximum scaling factor.
    :param tint_probability: Probability to apply tint.
    :param tint_intensity: Intensity of the tint.
    :param rotation_degree: Maximum rotation degree.
    :return: (updated_image, yolo_bboxes, dead_zones)
    """
    if dead_zones is None:
        dead_zones = []
    
    base_w, base_h = base_img.size
    yolo_bboxes = []
    
    # Gather all PNG files in icons_dir
    icon_files = sorted([
        f for f in glob.glob(os.path.join(icons_dir, "*.png"))
        if os.path.isfile(f)
    ])
    
    if not icon_files:
        print(f"No icon PNGs found in '{icons_dir}'.")
        return base_img, yolo_bboxes, dead_zones
    
    for i in range(num_icons):
        placed = False
        for attempt in range(max_tries):
            icon_path = random.choice(icon_files)
            class_id = extract_class_id(icon_path)
            if class_id is None:
                continue  # Skip icons with invalid class_id
            
            try:
                icon_img = Image.open(icon_path).convert("RGBA")
            except Exception as e:
                print(f"Error loading image '{icon_path}': {e}")
                continue
            
            # Random scale
            scale = random.uniform(min_scale, max_scale)
            new_size = (int(icon_img.width * scale), int(icon_img.height * scale))
            if new_size[0] < 10 or new_size[1] < 10:
                continue  # Skip too small icons
            try:
                icon_img = icon_img.resize(new_size, resample=Image.Resampling.BICUBIC)
            except AttributeError:
                # For older Pillow versions
                icon_img = icon_img.resize(new_size, resample=Image.BICUBIC)
            
            # Random rotation
            angle = random.uniform(-rotation_degree, rotation_degree)
            try:
                icon_img = icon_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            except AttributeError:
                # For older Pillow versions
                icon_img = icon_img.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # Random tint
            if random.random() < tint_probability:
                tint_color = tuple(random.randint(0, 255) for _ in range(3))
                icon_img = apply_color_tint(icon_img, tint_color, intensity=tint_intensity)
            
            # Determine possible placement area
            icon_w, icon_h = icon_img.size
            if icon_w >= base_w or icon_h >= base_h:
                continue  # Icon too big to fit
            
            # Random position
            max_x = base_w - icon_w
            max_y = base_h - icon_h
            if max_x <= 0 or max_y <= 0:
                continue  # No space to place
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Define bounding box
            box = (x, y, x + icon_w, y + icon_h)
            
            # Check for overlap
            if check_overlap(box, dead_zones):
                continue  # Overlaps, try another position
            
            # Place the icon
            base_img = overlay_icon(base_img, icon_img, x, y)
            
            # Update YOLO bounding box (normalized)
            x_center = (x + icon_w / 2) / base_w
            y_center = (y + icon_h / 2) / base_h
            width_norm = icon_w / base_w
            height_norm = icon_h / base_h
            
            yolo_bboxes.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width_norm,
                "height": height_norm
            })
            
            # Update dead zones
            dead_zones.append(box)
            placed = True
            break  # Move to next icon
        
        if not placed:
            print(f"Could not place icon {i+1}/{num_icons} after {max_tries} attempts.")
    
    return base_img, yolo_bboxes, dead_zones


def place_sprites(
    base_img,
    icons_dir="./icon_sprites",
    num_icons=5,
    max_tries=50,
    dead_zones=None,
    min_scale=0.1,
    max_scale=1.0,
    tint_probability=0.8,
    tint_intensity=0.3,
):
    """
    Places a specified number of icons onto the base image without overlapping.

    :param base_img: PIL.Image in RGBA mode.
    :param icons_dir: Directory containing icon PNGs.
    :param num_icons: Number of icons to place.
    :param max_tries: Maximum attempts to place each icon.
    :param dead_zones: List to track occupied regions.
    :param min_scale: Minimum scaling factor.
    :param max_scale: Maximum scaling factor.
    :param tint_probability: Probability to apply tint.
    :param tint_intensity: Intensity of the tint.
    :param rotation_degree: Maximum rotation degree.
    :return: (updated_image, yolo_bboxes, dead_zones)
    """
    if dead_zones is None:
        dead_zones = []
    
    base_w, base_h = base_img.size
    yolo_bboxes = []
    
    # Gather all PNG files in icons_dir
    icon_files = sorted([
        f for f in glob.glob(os.path.join(icons_dir, "*.png"))
        if os.path.isfile(f)
    ])
    
    if not icon_files:
        print(f"No icon PNGs found in '{icons_dir}'.")
        return base_img, yolo_bboxes, dead_zones
    
    for i in range(num_icons):
        placed = False
        for attempt in range(max_tries):
            icon_path = random.choice(icon_files)
            try:
                icon_img = Image.open(icon_path).convert("RGBA")
            except Exception as e:
                print(f"Error loading image '{icon_path}': {e}")
                continue
            
            # Random scale
            scale = random.uniform(min_scale, max_scale)
            new_size = (int(icon_img.width * scale), int(icon_img.height * scale))
            if new_size[0] < 10 or new_size[1] < 10:
                continue  # Skip too small icons
            try:
                icon_img = icon_img.resize(new_size, resample=Image.Resampling.BICUBIC)
            except AttributeError:
                # For older Pillow versions
                icon_img = icon_img.resize(new_size, resample=Image.BICUBIC)
            
      
            # Random tint
            if random.random() < tint_probability:
                tint_color = tuple(random.randint(0, 255) for _ in range(3))
                icon_img = apply_color_tint(icon_img, tint_color, intensity=tint_intensity)
            
            # Determine possible placement area
            icon_w, icon_h = icon_img.size
            if icon_w >= base_w or icon_h >= base_h:
                continue  # Icon too big to fit
            
            # Random position
            max_x = base_w - icon_w
            max_y = base_h - icon_h
            if max_x <= 0 or max_y <= 0:
                continue  # No space to place
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # Define bounding box
            box = (x, y, x + icon_w, y + icon_h)
            
            # Check for overlap
            if check_overlap(box, dead_zones):
                continue  # Overlaps, try another position
            
            # Place the icon
            base_img = overlay_icon(base_img, icon_img, x, y)
            
            
            # Update dead zones
            dead_zones.append(box)
            placed = True
            break  # Move to next icon
        
        if not placed:
            print(f"Could not place icon {i+1}/{num_icons} after {max_tries} attempts.")
    
    return base_img, yolo_bboxes, dead_zones

def main():
    num_images = 100 # Number of training images to generate
    output_images_dir = os.path.join("icon_training", "images", "val")
    output_labels_dir = os.path.join("icon_training", "labels", "val")
    
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    for i in range(num_images):
        # 1. Select a random background
        background_files = glob.glob(os.path.join("icon_background", "*.png"))
        if not background_files:
            print("No background PNGs found in './icon_background'.")
            return
        background_path = random.choice(background_files)
        try:
            base_img = Image.open(background_path).convert("RGBA")
        except Exception as e:
            print(f"Error loading background '{background_path}': {e}")
            continue
        
        dead_zones = []
        
        # 2. Place icons
        final_img, bboxes, dead_zones = place_icons(
            base_img,
            icons_dir="./icons",
            num_icons=5,
            dead_zones=dead_zones,
            min_scale=0.3,
            max_scale=1.0,
            tint_probability=0.8,
            tint_intensity=0.8,
            rotation_degree=0
        )

        final_img, bboxes, dead_zones = place_sprites(
            final_img,
            icons_dir="./icon_sprites",
            num_icons=1,
            dead_zones=dead_zones,
            min_scale=0.3,
            max_scale=1.0,
            tint_probability=0.5,
            tint_intensity=0.5,
        )
        
        # 3. Generate unique filenames
        unique_id = str(int(time.time() * 1000)) + f"_{i}"
        image_filename = f"final_image_{unique_id}.png"
        label_filename = f"final_image_{unique_id}.txt"
        
        # 4. Save the final image
        image_out_path = os.path.join(output_images_dir, image_filename)
        try:
            final_img.save(image_out_path)
            print(f"Saved image: {image_out_path}")
        except Exception as e:
            print(f"Error saving image '{image_out_path}': {e}")
            continue
        
        # 5. Save YOLO labels
        label_out_path = os.path.join(output_labels_dir, label_filename)
        try:
            with open(label_out_path, 'w') as f:
                for box in bboxes:
                    class_id = box['class_id']
                    x_center = box['x_center']
                    y_center = box['y_center']
                    width = box['width']
                    height = box['height']
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            print(f"Saved label: {label_out_path}")
        except Exception as e:
            print(f"Error saving label '{label_out_path}': {e}")
            continue
        
        print(f"Image {i+1}/{num_images} generated.\n")
    
    print("All images and labels have been generated.")

if __name__ == "__main__":
    main()
