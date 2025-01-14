import os
import glob
import random
import time

from PIL import Image, ImageEnhance


from PIL import Image

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
    # Only the RGB channels are blended; alpha remains untouched
    tinted_rgb = Image.blend(image.convert('RGB'), Image.new('RGB', image.size, tint_color), intensity)
    
    # Reattach the original alpha channel
    tinted_image = Image.merge('RGBA', (*tinted_rgb.split(), a))
    
    return tinted_image


def overlay_tower(base_img: Image.Image, tower_path: str, x: int, y: int, class_id: int = 999):
    """
    Overlays the tower image onto the base image at top-left (x, y).
    - If x or y is negative or the tower extends beyond the base,
      the tower is partially (or fully) clipped automatically by Pillow.
    - Returns (updated_image, yolo_box_dict).

    YOLO bounding box data is returned for the visible region of the tower
    that lies within the base image boundaries. If the tower is completely
    out of bounds, width/height may be zero.

    :param base_img:  PIL Image in RGBA mode (the base arena or background).
    :param tower_path: Path to the tower PNG (relative to this .py file).
    :param x:         X-coordinate of tower's top-left corner (can be negative).
    :param y:         Y-coordinate of tower's top-left corner (can be negative).
    :param class_id:  Integer class ID for YOLO (default=999 if you like).
    :return: (combined_image, yolo_box)
             where yolo_box is a dict with keys:
               {
                 'class_id': class_id,
                 'x_center': float,
                 'y_center': float,
                 'width': float,
                 'height': float
               }
    """
    tower_img = Image.open(tower_path).convert("RGBA")
    base_w, base_h = base_img.size
    tower_w, tower_h = tower_img.size

    # 1) Create a blank (transparent) overlay the same size as the base
    overlay_layer = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
    
    # 2) Paste the tower onto this overlay at (x, y).
    #    Any portion outside [0..base_w, 0..base_h] is automatically clipped by Pillow.
    overlay_layer.paste(tower_img, (x, y))

    # 3) Alpha-composite overlay_layer on top of base_img
    combined = Image.alpha_composite(base_img, overlay_layer)

    # 4) Compute the visible bounding box of the tower within the base image.
    #    We clamp coordinates so that if part of the tower is offscreen, 
    #    only the visible portion is included in the bounding box.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + tower_w, base_w)
    y2 = min(y + tower_h, base_h)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        # The tower is completely out of frame or has zero visible area
        yolo_box = {
            'class_id': class_id,
            'x_center': 0.0,
            'y_center': 0.0,
            'width': 0.0,
            'height': 0.0
        }
    else:
        # 5) Convert to YOLO format (normalized [0..1])
        x_center = (x1 + box_w/2) / base_w
        y_center = (y1 + box_h/2) / base_h
        width_norm = box_w / base_w
        height_norm = box_h / base_h

        yolo_box = {
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width_norm,
            'height': height_norm
        }

    return combined, yolo_box


def overlay_troop_and_clamp(base_img, troop_img, x, y, 
                            dead_zones=None, overlap_threshold=0.3):
    """
    Overlays a troop image onto base_img at top-left (x, y), unless the troop
    overlaps a dead zone beyond a certain threshold.

    1. x and y must be >= 0. If (x + troop_width) > base_width (or similarly for y),
       clamp so the troop fits within the base image.
    2. If the troop's bounding box overlaps any 'dead zone' more than `overlap_threshold`
       fraction of the troop's area, skip placing (return (0,0,0,0) for the bbox).
    3. Returns:
       - updated_image, (x_center_norm, y_center_norm, width_norm, height_norm)
         in YOLO format, or (0,0,0,0) if skipped.

    :param base_img:   PIL Image in RGBA mode (the "base" or background).
    :param troop_path: Path to the troop PNG (relative to this .py).
    :param x:          Desired top-left X coordinate (clamped if needed).
    :param y:          Desired top-left Y coordinate (clamped if needed).
    :param dead_zones: List of [zx, zy, zw, zh] rectangles disallowed beyond overlap_threshold.
    :param overlap_threshold: Fraction [0..1], skip if (overlap_area / troop_area) > threshold.
    :return: (updated_img, (x_center_norm, y_center_norm, width_norm, height_norm))
    """
    if dead_zones is None:
        dead_zones = []

    # Convert base image to RGBA just in case
    base_img = base_img.convert("RGBA")
    troop_img = troop_img.convert("RGBA")

    base_w, base_h = base_img.size
    troop_w, troop_h = troop_img.size

    # 1) Clamp x, y to fit within the base
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    if x + troop_w > base_w:
        x = base_w - troop_w
    if y + troop_h > base_h:
        y = base_h - troop_h

    # Troop bounding box (in pixels)
    troop_x1, troop_y1 = x, y
    troop_x2, troop_y2 = x + troop_w, y + troop_h

    troop_area = troop_w * troop_h  # entire bounding box of the troop

    # Function to compute overlap area
    def overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """
        Returns the area of overlap (in pixels) between box A and box B.
        Boxes are [x1, y1, x2, y2].
        """
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0  # no overlap
        return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # 2) Check overlap with each dead zone
    for (zx, zy, zw, zh) in dead_zones:
        zone_x1, zone_y1 = zx, zy
        zone_x2, zone_y2 = zx + zw, zy + zh

        # Overlap area in pixels
        inter_area = overlap_area(troop_x1, troop_y1, troop_x2, troop_y2,
                                  zone_x1, zone_y1, zone_x2, zone_y2)

        if inter_area > 0:
            fraction_overlap = inter_area / troop_area

            # If fraction of troop area that overlaps is > overlap_threshold, skip
            if fraction_overlap > overlap_threshold:
                print(f"Skipping troop: Overlap with dead zone > {overlap_threshold*100:.1f}%")
                return base_img, (0.0, 0.0, 0.0, 0.0)

    # 3) If no zone overlap above threshold, place the troop
    overlay_layer = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
    overlay_layer.paste(troop_img, (x, y))
    updated_img = Image.alpha_composite(base_img, overlay_layer)

    # 4) Compute YOLO bounding box
    x_center_norm = (x + troop_w / 2) / base_w
    y_center_norm = (y + troop_h / 2) / base_h
    width_norm    = troop_w / base_w
    height_norm   = troop_h / base_h

    bbox_yolo = (x_center_norm, y_center_norm, width_norm, height_norm)

    return updated_img, bbox_yolo

def place_random_troops(
    base_img,
    troops_dir="./troops",
    padding=50,
    min_troops=0,
    max_troops=2,
    max_tries=10,
    dead_zones=None,
    min_scale=0.5,
    max_scale=1.0,
    color_jitter=0.9,
    mirror_probability=0.5,
    tint_probability=0.8,      # Probability to apply a tint
    tint_intensity=0.6         # Intensity of the tint
):
    """
    1) Finds all subfolders in `troops_dir` named like 'troop_XX' => class_id = XX
    2) For each subfolder, picks a random integer in [min_troops..max_troops].
    3) For each troop: pick a random PNG, apply random scale & color shift & tint.
    4) Randomly pick (x,y), call overlay_troop_and_clamp up to `max_tries`.
    5) On success, add bounding box to dead_zones (so no overlap for next troops).
    6) Return the updated image + a list of YOLO bboxes.

    :param base_img:    PIL Image for the background.
    :param troops_dir:  Directory with subfolders 'troop_<classid>'.
    :param padding:     Min. border distance.
    :param min_troops, max_troops: range of how many troops from each subfolder.
    :param max_tries:   times to try placing before giving up.
    :param dead_zones:  list of [x, y, w, h] rectangles to avoid.
    :param min_scale, max_scale:  scale range for troop resizing.
    :param color_jitter: float, how much brightness/color can vary (0.2 => ±20%).
    :param mirror_probability: float, probability to mirror the troop image.
    :param tint_probability: float, probability to apply a color tint.
    :param tint_intensity: float, intensity of the color tint.
    """
    if dead_zones is None:
        dead_zones = []

    base_w, base_h = base_img.size
    yolo_bboxes = []

    # Subfolders => classes
    subfolders = sorted([
        d for d in os.listdir(troops_dir)
        if os.path.isdir(os.path.join(troops_dir, d))
    ])

    updated_img = base_img.copy()  # Make a copy to avoid modifying the original

    for folder_name in subfolders:
        subfolder_path = os.path.join(troops_dir, folder_name)

        # Parse class ID
        try:
            class_id_str = folder_name.split("_", 1)[-1]
            class_id = int(class_id_str)
        except ValueError:
            print(f"Warning: Could not parse class_id from folder '{folder_name}'. Skipping.")
            continue

        # PNG files in subfolder
        png_files = sorted([
            f for f in glob.glob(os.path.join(subfolder_path, "*.png"))
            if os.path.isfile(f)
        ])
        if not png_files:
            continue

        # How many troops to place from this subfolder
        count_to_place = random.randint(min_troops, max_troops)

        for _ in range(count_to_place):
            troop_path = random.choice(png_files)

            placed_successfully = False

            for attempt in range(max_tries):
                # 1) Load the troop
                try:
                    troop_img = Image.open(troop_path).convert("RGBA")
                except Exception as e:
                    print(f"Error loading image '{troop_path}': {e}")
                    break  # Skip this troop

                if random.random() < mirror_probability:
                    troop_img = troop_img.transpose(Image.FLIP_LEFT_RIGHT)
                # 2) Random scale
                scale_factor = random.uniform(min_scale, max_scale)
                new_w = int(troop_img.width * scale_factor)
                new_h = int(troop_img.height * scale_factor)
                if new_w < 1 or new_h < 1:
                    # Extremely small => skip
                    continue
                troop_img = troop_img.resize((new_w, new_h), Image.LANCZOS)

                # 3) Random color shift: brightness & color
                brightness_factor = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)
                color_factor      = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)

                # Apply brightness shift
                troop_img = ImageEnhance.Brightness(troop_img).enhance(brightness_factor)
                # Apply color shift (affects saturation)
                troop_img = ImageEnhance.Color(troop_img).enhance(color_factor)

                # 4) Apply color tint (red, blue, or random)
                if random.random() < tint_probability:
                    tint_choice = random.choice(['red', 'blue', 'random'])
                    if tint_choice == 'red':
                        tint_color = (255, 0, 0)  # Red
                    elif tint_choice == 'blue':
                        tint_color = (0, 0, 255)  # Blue
                    elif tint_choice == 'random':
                        tint_color = tuple(random.randint(0, 255) for _ in range(3))  # Random color
                    else:
                        tint_color = None  # No tint

                    if tint_color:
                        troop_img = apply_color_tint(troop_img, tint_color, intensity=tint_intensity)

                # 5) Random (x, y)
                # Ensure the troop fits within the base image considering padding
                max_x = base_w - new_w - padding
                max_y = base_h - new_h - padding
                if max_x < padding or max_y < padding:
                    # Not enough space to place the troop with the given padding
                    continue

                x = random.randint(padding, max_x)
                y = random.randint(padding, max_y)

                # 6) Attempt overlay
                maybe_img, (x_center, y_center, w_norm, h_norm) = overlay_troop_and_clamp(
                    updated_img,
                    troop_img,
                    x,
                    y,
                    dead_zones=dead_zones
                )

                # 7) Check success
                if (x_center, y_center, w_norm, h_norm) != (0.0, 0.0, 0.0, 0.0):
                    # Success => update the base image
                    updated_img = maybe_img
                    yolo_bboxes.append({
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": w_norm,
                        "height": h_norm
                    })

                    # Convert YOLO coords back to pixel coords => add to dead_zones
                    troop_w_pixels = int(w_norm * base_w)
                    troop_h_pixels = int(h_norm * base_h)
                    troop_x_pixels = int(x_center * base_w - troop_w_pixels / 2)
                    troop_y_pixels = int(y_center * base_h - troop_h_pixels / 2)
                    # Clamp
                    troop_x_pixels = max(0, troop_x_pixels)
                    troop_y_pixels = max(0, troop_y_pixels)

                    dead_zones.append([
                        troop_x_pixels,
                        troop_y_pixels,
                        troop_w_pixels,
                        troop_h_pixels
                    ])

                    placed_successfully = True
                    break  # Done placing this troop

            if not placed_successfully:
                print(f"Skipping troop '{troop_path}' after {max_tries} attempts.")

    return updated_img, yolo_bboxes, dead_zones


def placeTower(base_img, tower_path, yolo_bboxes, class_id, dead_zones=None, padding=50, min_scale=0.6,
    max_scale=1.4,color_jitter = 0.4,  max_tries = 10):
    base_w, base_h = base_img.size
    updated_img = base_img
    for attempt in range(max_tries):
        # 1) Load the troop
        troop_img = Image.open(tower_path).convert("RGBA")

        # 2) Random scale
        scale_factor = random.uniform(min_scale, max_scale)
        new_w = int(troop_img.width * scale_factor)
        new_h = int(troop_img.height * scale_factor)
        if new_w < 1 or new_h < 1:
            # extremely small => skip
            continue
        troop_img = troop_img.resize((new_w, new_h), Image.LANCZOS)

        # 3) Random color shift: brightness & color
        brightness_factor = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)
        color_factor      = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)

        # Apply brightness shift
        troop_img = ImageEnhance.Brightness(troop_img).enhance(brightness_factor)
        # Apply color shift (affects saturation)
        troop_img = ImageEnhance.Color(troop_img).enhance(color_factor)

        # 4) Random (x, y)
        x = random.randint(padding, max(padding, base_w - padding))
        y = random.randint(padding, max(padding, base_h - padding))

        # 5) Attempt overlay
        maybe_img, (x_center, y_center, w_norm, h_norm) = overlay_troop_and_clamp(
            updated_img,
            troop_img,
            x,
            y,
            dead_zones=dead_zones
        )

        # 6) Check success
        if (x_center, y_center, w_norm, h_norm) != (0.0, 0.0, 0.0, 0.0):
            # success => update the base image
            updated_img = maybe_img
            yolo_bboxes.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": w_norm,
                "height": h_norm
            })

            # Convert YOLO coords back to pixel coords => add to dead_zones
            troop_w_pixels = int(w_norm * base_w)
            troop_h_pixels = int(h_norm * base_h)
            troop_x_pixels = int(x_center * base_w - troop_w_pixels / 2)
            troop_y_pixels = int(y_center * base_h - troop_h_pixels / 2)
            # clamp
            troop_x_pixels = max(0, troop_x_pixels)
            troop_y_pixels = max(0, troop_y_pixels)

            dead_zones.append([
                troop_x_pixels,
                troop_y_pixels,
                troop_w_pixels,
                troop_h_pixels
            ])

            placed_successfully = True
            break  # done placing this troop

    if not placed_successfully:
        print(f"Skipping troop '{tower_path}' after {max_tries} attempts.")
    return updated_img, yolo_bboxes, dead_zones


def place_random_decoy_sprites(
    base_img,
    decoys_dir="./sprites",
    padding=0,
    min_decoys=10,
    max_decoys=20,
    max_tries=10,
    dead_zones=None,
    min_scale=0.3,
    max_scale=1.0,
    color_jitter=0.5
):
    """
    Places a random number of decoy sprites from 'decoys_dir' onto 'base_img',
    respecting 'dead_zones' (i.e., won't place if top-left is in a dead zone).
    These decoys do NOT produce labels (they're just background distractions).

    1) For each decoy sprite, we randomly scale it (min_scale..max_scale).
    2) Randomly apply color shifts (brightness and saturation).
    3) Randomly pick (x, y) within [padding..base_dim - padding].
    4) Call overlay_decoy_and_clamp(...) (or a similar function) to place it.
       If it fails (returns skip) we retry up to max_tries times.
    5) If placed successfully, we add the bounding box to 'dead_zones' so nothing else
       can overlap that decoy area. We do NOT produce YOLO labels.

    Returns:
        updated_img  (the final image with decoys placed)
    """
    if dead_zones is None:
        dead_zones = []

    base_w, base_h = base_img.size
    updated_img = base_img

    # Gather all PNG files in decoys_dir
    decoy_files = sorted([
        f for f in glob.glob(os.path.join(decoys_dir, "*.png"))
        if os.path.isfile(f)
    ])

    if not decoy_files:
        print(f"No decoy PNGs found in '{decoys_dir}'. Skipping decoys.")
        return updated_img

    # Decide how many decoys to place
    num_decoys = random.randint(min_decoys, max_decoys)

    for _ in range(num_decoys):
        
        placed_successfully = False

        for attempt in range(max_tries):
            # 1) Pick a random decoy PNG
            decoy_path = random.choice(decoy_files)

            # 2) Load and scale
            decoy_img = Image.open(decoy_path).convert("RGBA")
            scale_factor = random.uniform(min_scale, max_scale)
            new_w = int(decoy_img.width * scale_factor)
            new_h = int(decoy_img.height * scale_factor)
            if new_w < 1 or new_h < 1:
                continue  # too small, skip this attempt

            decoy_img = decoy_img.resize((new_w, new_h), Image.LANCZOS)

            # 3) Color shift
            brightness_factor = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)
            color_factor      = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)

            decoy_img = ImageEnhance.Brightness(decoy_img).enhance(brightness_factor)
            decoy_img = ImageEnhance.Color(decoy_img).enhance(color_factor)

            # 4) Random (x, y)
            x = random.randint(padding, max(padding, base_w - padding))
            y = random.randint(padding, max(padding, base_h - padding))

            # 5) Attempt overlay (top-left approach, no YOLO bounding box)
            maybe_img, tt = overlay_troop_and_clamp(
                updated_img,
                decoy_img,
                x,
                y,
                dead_zones,
                overlap_threshold=0.1
            )
            success = True
            if tt == (0.0, 0.0, 0.0, 0.0):
                success = False
            if success:
                # Update image
                updated_img = maybe_img

                # Add to dead_zones if you want to prevent overlap
                decoy_x = x
                decoy_y = y
                decoy_w = decoy_img.width
                decoy_h = decoy_img.height

                dead_zones.append([
                    decoy_x,
                    decoy_y,
                    decoy_w,
                    decoy_h
                ])

                placed_successfully = True
                break

        if not placed_successfully:
            print(f"Skipping decoy sprite after {max_tries} attempts.")
        print(f"Placed {num_decoys} decoys.")
    return updated_img

def main():
    number_of_data = 490
    for a in range(number_of_data):
        # 1) Randomly choose an arena base
        arena_files = glob.glob("./arenas/*.png")
        if not arena_files:
            print("No PNG files found in './arenas'.")
            return
        
        base_path = random.choice(arena_files)
        base_img = Image.open(base_path).convert("RGBA")
        dead_zones = [
            # [416,0, 169, 269],
            # [416,991, 169, 269],
            # [175,160, 171, 219],
            # [687,160, 171, 219],
            # [175,878, 171, 219],
            # [687,878, 171, 219]
        ]
        # 2) Place random troops
        #    (Ensure you have overlay_troop_and_clamp & place_random_troops imported/defined.)
        result_img, bboxes, dead_zones = place_random_troops(base_img, troops_dir="./troops", dead_zones=dead_zones)

        # 3) Overlay towers (with assigned class_ids)
        #    Note: Each call returns (updated_img, yolo_box), so append it to `bboxes`.
        # result_img, bboxes, dead_zones = placeTower(result_img, "./towers/king/enemy/0.png", bboxes, class_id=2, dead_zones=dead_zones)

        # result_img, bboxes, dead_zones = placeTower(result_img, "./towers/king/friendly/0.png", bboxes, class_id=0, dead_zones=dead_zones)

        # result_img, bboxes, dead_zones = placeTower(result_img, "./towers/regular/friendly/0.png", bboxes, class_id=1, dead_zones=dead_zones)
        # result_img, bboxes, dead_zones = placeTower(result_img, "./towers/regular/friendly/0.png", bboxes, class_id=1, dead_zones=dead_zones)

        # result_img, bboxes, dead_zones = placeTower(result_img, "./towers/regular/enemy/0.png", bboxes, class_id=3, dead_zones=dead_zones)
        # result_img, bboxes, dead_zones = placeTower(result_img, "./towers/regular/enemy/0.png", bboxes, class_id=3, dead_zones=dead_zones)

        result_img = place_random_decoy_sprites(result_img, decoys_dir="./sprites", dead_zones=dead_zones)
        # result_img, tower_box = overlay_tower(result_img, "./towers/king/friendly/0.png", x=416, y=991, class_id=0)
        # bboxes.append(tower_box)

        # result_img, tower_box = overlay_tower(result_img, "./towers/regular/enemy/0.png", x=175, y=160, class_id=3)
        # bboxes.append(tower_box)

        # result_img, tower_box = overlay_tower(result_img, "./towers/regular/enemy/0.png", x=687, y=160, class_id=3)
        # bboxes.append(tower_box)

        # result_img, tower_box = overlay_tower(result_img, "./towers/regular/friendly/0.png", x=175, y=878, class_id=1)
        # bboxes.append(tower_box)

        # result_img, tower_box = overlay_tower(result_img, "./towers/regular/friendly/0.png", x=687, y=878, class_id=1)
        # bboxes.append(tower_box)

        # 4) Create output directories if needed
        # os.makedirs("out/images", exist_ok=True)
        # os.makedirs("out/labels", exist_ok=True)

        # 5) Generate a unique name for this sample.
        #    You could also use a simple counter or a timestamp.
        #    For example, let's use the time.time() float as part of the filename:
        unique_id = str(int(time.time() * 1000))  # or you could use uuid, etc.

        image_filename = f"final_arena_{unique_id}.png"
        label_filename = f"final_arena_{unique_id}.txt"

        # 6) Save the final composited image
        image_out_path = os.path.join("out", "images", "train", image_filename)
        result_img.save(image_out_path)
        print(f"Final image saved to: {image_out_path}")

        # 7) Write YOLO label file
        label_out_path = os.path.join("out", "labels", "train", label_filename)
        with open(label_out_path, 'w') as f:
            for box in bboxes:
                class_id = box['class_id']
                x_center = box['x_center']
                y_center = box['y_center']
                width_   = box['width']
                height_  = box['height']
                # Each line: class x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_:.6f} {height_:.6f}\n")

        print(f"Label file saved to: {label_out_path}")
        print("Done.")

if __name__ == "__main__":
    main()
