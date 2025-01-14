import cv2

def draw_yolo_bboxes(image_path, label_path):
    """
    Loads an image, parses its YOLO label file, and displays the image
    with bounding boxes drawn.
    """
    # 1) Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2) Get image dimensions
    h, w, _ = image.shape

    # 3) Read the label file line by line
    try:
        with open(label_path, 'r') as f:
            lines = f.read().strip().split('\n')
    except FileNotFoundError:
        print(f"Error: Could not find label file at {label_path}")
        return

    # 4) Parse and draw boxes
    for line in lines:
        if not line.strip():
            continue  # skip empty lines

        # Each line should have 5 elements: class_id, x_center, y_center, width, height
        parts = line.split()
        if len(parts) != 5:
            print(f"Warning: skipping malformed line: {line}")
            continue

        class_id, x_center_norm, y_center_norm, width_norm, height_norm = parts
        # Convert from string to float
        class_id = int(class_id)
        x_center_norm = float(x_center_norm)
        y_center_norm = float(y_center_norm)
        width_norm = float(width_norm)
        height_norm = float(height_norm)

        # Convert normalized coords [0..1] to pixel coords
        x_center = int(x_center_norm * w)
        y_center = int(y_center_norm * h)
        box_width = int(width_norm * w)
        box_height = int(height_norm * h)

        # Calculate box corners
        x1 = x_center - box_width // 2
        y1 = y_center - box_height // 2
        x2 = x_center + box_width // 2
        y2 = y_center + box_height // 2

        # Draw rectangle
        # (Use a color for the box, e.g., green = (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optionally, draw class_id text near the box
        cv2.putText(image, f"Class: {class_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5) Display the resulting image
    cv2.imshow("YOLO BBoxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # Provide the path to an image and its corresponding YOLO label file
    image_path = "./out/images/final_arena_1735910114133.png"
    label_path = "./out/labels/final_arena_1735910114133.txt"
    draw_yolo_bboxes(image_path, label_path)
