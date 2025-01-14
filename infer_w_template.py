import torch
import numpy as np
import cv2
from mss import mss
import logging
import warnings
import time
import os

# ---------------------------
# Configuration
# ---------------------------

# Suppress all warnings (optional)
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='number_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load your YOLOv5 model with AutoShape
# Ensure you're using a compatible YOLOv5 version
# If YOLOv5 is not installed via torch.hub, ensure the path and settings are correct
model = torch.hub.load(
    './yolov5',  # Local path to YOLOv5 repository
    'custom',
    path='./yolov5/runs/train/yolov5s_w_sprites/weights/best.pt',  # Path to your custom YOLOv5 model weights
    source='local'
)
model2 = torch.hub.load(
    './yolov5',  # Local path to YOLOv5 repository
    'custom',
    path='./yolov5/runs/train/finetune_new2/weights/best.pt',  # Path to your custom YOLOv5 model weights
    source='local'
)
# Initialize MSS screen capture
sct = mss()

# Define a bounding box for screen capture (adjust to your setup)
monitor = {
    "top": 50,      # Y-position of the top edge (absolute screen coordinate)
    "left": 0,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 1300  # Capture height
}
yolo_monitor = {
    "top": 0,      # Y-position of the top edge (absolute screen coordinate)
    "left": 0,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 1050  # Capture height
}

card_monitor = {
    "top": 1050,      # Y-position of the top edge (absolute screen coordinate)
    "left": 0,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 250  # Capture height
}

# Template Matching Configuration
TEMPLATE_DIR = './num'  # Directory containing digit templates named '0.png' to '9.png'

MATCH_METHOD = cv2.TM_CCOEFF_NORMED
NMS_THRESHOLD = 0.5  # Non-Maximum Suppression threshold
MAX_GAP = 5          # Maximum gap between digits to group them into a number
MAX_Y_GAP = 5
# Define per-digit matching thresholds
DIGIT_THRESHOLDS = {
    '0': 0.75,
    '1': 0.85,  # Higher threshold for '1' to reduce false positives
    '2': 0.75,
    '3': 0.75,
    '4': 0.8,
    '5': 0.75,
    '6': 0.75,
    '7': 0.8,
    '8': 0.8,
    '9': 0.8
}

# Health Bar Configuration
# Adjust these parameters based on your health bar's appearance and position
health_bars = [
    {
        "name": "let_health",
        "top": 125,
        "left": 75,
        "width": 200,
        "height": 100,
    },
    {
        "name": "ret_health",
        "top": 125,
        "left": 450,
        "width": 200,
        "height": 100,
    },
    {
        "name": "lft_health",
        "top": 750,
        "left": 75,
        "width": 200,
        "height": 100,
    },
    {
        "name": "rft_health",
        "top": 750,
        "left": 450,
        "width": 200,
        "height": 100,
    },
    {
        "name": "cfk_health",
        "top": 950,
        "left": 275,
        "width": 200,
        "height": 100,
    },
    {
        "name": "cek_health",
        "top": 0,
        "left": 275,
        "width": 200,
        "height": 100,
    },
]

elixir_points = [
    i * 53 + 222 for i in reversed(range(10))
]
elixir_color = [
    208, 32, 216
]

# ---------------------------
# Utility Functions
# ---------------------------

def load_digit_templates(template_dir):
    """
    Loads and preprocesses digit templates from the specified directory without resizing.

    :param template_dir: Path to the folder containing digit templates named '0.png' to '9.png'.
    :return: Dictionary mapping digit strings ('0'-'9') to their template images.
    """
    templates = {}
    for digit in range(10):
        filename = f"{digit}.png"
        path = os.path.join(template_dir, filename)
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Warning: Template '{filename}' not found in '{template_dir}'. Skipping.")
            continue
        # Optional: Apply Gaussian Blur to reduce noise
        template_blurred = cv2.GaussianBlur(template, (3, 3), 0)
        templates[str(digit)] = template_blurred
    return templates

def preprocess_image(image):
    """
    Preprocesses the image for template matching.

    :param image: Input BGR image.
    :return: Preprocessed grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optional: Apply Gaussian Blur to reduce noise
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def draw_centered_rectangles(frame, rect_size=20, color=(0, 0, 0), thickness=2, tolerance = 80):
    """
    Draws black rectangles centered around the specified points on the frame.

    :param frame: The image/frame on which to draw.
    :param points: List of (x, y) tuples representing the points.
    :param rect_size: Size of the rectangle (width and height). Defaults to 20 pixels.
    :param color: Rectangle color in BGR. Defaults to black (0, 0, 0).
    :param thickness: Thickness of the rectangle border. If -1, the rectangle is filled.
    """
    # print("is it here. rect_size: ", rect_size)
    half_size = rect_size // 2
    y = 245
    elixir = 10
    for idx, x in enumerate(elixir_points):
        met = True
        px_color = frame[y,x]
        # print(px_color)
        for i in range(3):  # B, G, R channels
            if not (elixir_color[i] - tolerance <= px_color[i] <= elixir_color[i] + tolerance):       
                top_left = (x - half_size, y - half_size)
                bottom_right = (x + half_size, y + half_size)

                cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                # Optionally, label the point number
                cv2.putText(frame, f"{idx+1}", (x - half_size, y + half_size + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                elixir -= 1
                met = False
                break
        if met:  
            break
    cv2.putText(frame, f"{elixir}", (x - half_size, y - half_size - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return elixir


def perform_template_matching(preprocessed_roi, templates, method=MATCH_METHOD, digit_thresholds=DIGIT_THRESHOLDS):
    """
    Performs template matching for each digit template on the ROI without resizing.

    :param preprocessed_roi: Preprocessed grayscale ROI image.
    :param templates: Dictionary of digit templates.
    :param method: Template matching method.
    :param digit_thresholds: Dictionary mapping digits to their specific thresholds.
    :return: List of detected digits with their positions and matched digit.
    """
    detections = []

    for digit, template in templates.items():
        template_height, template_width = template.shape
        res = cv2.matchTemplate(preprocessed_roi, template, method)
        threshold = digit_thresholds.get(digit, MATCH_METHOD)  # Use specific threshold if defined

        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):  # Switch columns and rows
            detections.append({
                'digit': digit,
                'position': pt,
                'score': res[pt[1], pt[0]],
                'size': (template_width, template_height)
            })

    return detections

def non_max_suppression(detections, overlap_thresh=NMS_THRESHOLD):
    """
    Applies Non-Maximum Suppression to eliminate overlapping detections.

    :param detections: List of detections with 'position' and 'size'.
    :param overlap_thresh: Threshold for overlapping areas.
    :return: List of filtered detections.
    """
    if not detections:
        return []

    # Initialize lists for bounding boxes and corresponding scores
    boxes = []
    scores = []

    for det in detections:
        x, y = det['position']
        w, h = det['size']
        boxes.append([x, y, x + w, y + h])
        scores.append(det['score'])

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Convert to float type
    boxes = boxes.astype(float)

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes and sort by scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        pick.append(i)

        # Compute intersection
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute width and height of the intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / areas[order[1:]]

        # Keep indexes with overlap less than threshold
        inds = np.where(overlap <= overlap_thresh)[0]
        order = order[inds + 1]

    # Return the filtered detections
    filtered_detections = [detections[i] for i in pick]
    return filtered_detections

def filter_detections(detections):
    """
    Applies additional filtering to detections, especially for digit '1'.

    :param detections: List of detections with 'digit', 'position', 'score', and 'size'.
    :return: Filtered list of detections.
    """
    filtered = []
    for det in detections:
        if det['digit'] == '1':
            w, h = det['size']
            aspect_ratio = h / w if w != 0 else 0
            # Example condition: '1' should be taller than a certain aspect ratio
            if aspect_ratio < 2.0:  # Adjust based on your template's aspect ratio
                continue  # Skip detections that don't meet the aspect ratio
        filtered.append(det)
    return filtered

def group_digits(detections, max_gap=MAX_GAP, max_Y_gap=MAX_Y_GAP):
    """
    Groups detected digits into numbers based on their x-coordinates.

    :param detections: List of detections with 'position' and 'digit'.
    :param max_gap: Maximum gap between digits to consider them part of the same number.
    :return: Recognized number as a string.
    """
    if not detections:
        return ""

    # Sort detections left to right based on x-coordinate
    detections = sorted(detections, key=lambda x: x['position'][0])

    numbers = []
    current_number = detections[0]['digit']
    last_x, last_y = detections[0]['position']
    last_w, last_h = detections[0]['size']

    for det in detections[1:]:
        x, y = det['position']
        w, h = det['size']
        gap = x - (last_x + last_w)
        y_gap = y - (last_y )

        if gap <= max_gap and y_gap <= max_Y_gap:
            current_number += det['digit']
        else:
            numbers.append(current_number)
            current_number = det['digit']

        last_x, last_y = x, y
        last_w, last_h = w, h

    numbers.append(current_number)
    return numbers

def annotate_detections(frame, detections, recognized_numbers):
    """
    Annotates detected digits and recognized numbers on the frame.

    :param frame: The frame to annotate.
    :param detections: List of filtered detections.
    :param recognized_numbers: List of recognized numbers.
    """
    for det in detections:
        x, y = det['position']
        w, h = det['size']
        digit = det['digit']
        score = det['score']

        # Draw rectangle around detected digit
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put the digit label above the rectangle
        cv2.putText(frame, digit, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Overlay the recognized numbers on the frame
    y_position = 30
    for idx, number in enumerate(recognized_numbers):
        cv2.putText(frame, f"Number {idx + 1}: {number}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        y_position += 30

# ---------------------------
# Main Loop
# ---------------------------

frame_count = 0
start_time = time.time()

# Load digit templates
digit_templates = load_digit_templates(TEMPLATE_DIR)
if not digit_templates:
    print("Error: No digit templates loaded. Please check the './num/' directory.")
    exit()
print(f"Loaded digit templates: {list(digit_templates.keys())}")

while True:
    try:
        frame_count += 1

        # Grab screen
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        yolo_top = yolo_monitor["top"]
        yolo_left = yolo_monitor["left"]
        yolo_width = yolo_monitor["width"]
        yolo_height = yolo_monitor["height"]
        yolo_frame = frame_bgr[yolo_top:yolo_top + yolo_height, yolo_left:yolo_left + yolo_width]

        # Create a pure copy for template matching (no YOLO annotations)
        pure_frame = yolo_frame.copy()

        # Run YOLO inference on the main frame
        results = model(yolo_frame, size=640)  # AutoShape handles resizing

        # Render YOLO detections on a separate frame
        # results.render()  # This modifies results.imgs in place
        img = results.render()

        
        annotated_frame = np.squeeze(img)  # Ensure it's a 2D image

        # Initialize list to collect all recognized numbers
        all_recognized_numbers = []

        # Iterate over predefined health bars
        for health_bar in health_bars:
            top = health_bar["top"]
            left = health_bar["left"]
            width = health_bar["width"]
            height = health_bar["height"]

            # Extract the ROI for the health bar from the pure frame
            roi = pure_frame[top:top + height, left:left + width]

            # Check if ROI is valid
            if roi.size == 0:
                logging.error(f"ROI for {health_bar['name']} is empty. Check monitor coordinates.")
                print(f"Error: ROI for {health_bar['name']} is empty. Check monitor coordinates.")
                continue

            # Preprocess the ROI
            preprocessed_roi = preprocess_image(roi)
            # preprocessed_roi = roi
            # Perform template matching
            detections = perform_template_matching(preprocessed_roi, digit_templates)
            detections = filter_detections(detections)
            filtered_detections = non_max_suppression(detections)

            # Adjust positions relative to the entire frame
            for det in filtered_detections:
                det['position'] = (det['position'][0] + left, det['position'][1] + top)

            # Group detected digits into numbers
            recognized_number = group_digits(filtered_detections)
            if recognized_number:
                all_recognized_numbers.append(recognized_number)

            # Annotate detections on the annotated frame
            annotate_detections(annotated_frame, filtered_detections, [recognized_number])

            # Log the recognized number
            if recognized_number:
                logging.info(f"{health_bar['name']}: Recognized Number: {recognized_number}")
                print(f"{health_bar['name']}: Recognized Number: {recognized_number}")

        # Show the resulting frame with YOLO boxes and digit detections
        cv2.imshow("YOLOv5 Detections with Number Monitoring", annotated_frame)
        card_frame = frame_bgr[card_monitor["top"]:card_monitor["top"] + card_monitor["height"], card_monitor["left"]:card_monitor["left"] + card_monitor["width"]]
        deck_results = model2(card_frame, size=200)  # AutoShape handles resizing
        deck_results.render()  # This modifies results.imgs in place
        annotated_deck = np.squeeze(deck_results.render())  # Ensure it's a 2D image
        draw_centered_rectangles(card_frame)
        cv2.imshow("Card Monitor", card_frame)
        cv2.imshow("annoted deck", annotated_deck)
        # Calculate and display FPS every 60 frames
        if frame_count % 60 == 0:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        break
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")  # Optional: Print error to console for immediate feedback

# Cleanup
cv2.destroyAllWindows()
