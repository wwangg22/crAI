import torch
import numpy as np
import cv2
from mss import mss
import logging
import warnings
import time

# ---------------------------
# Configuration
# ---------------------------

# Suppress all warnings (optional)
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='health_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load your YOLOv5 model with AutoShape
# Ensure you're using a compatible YOLOv5 version
model = torch.hub.load(
    './yolov5',  # Local path to YOLOv5 repository
    'custom',
    path='./yolov5/runs/train/yolov5s_w_sprites/weights/best.pt',
    source='local'
)

# Initialize MSS screen capture
sct = mss()

# Define a bounding box for LDPlayer’s window (example coords, adjust to your setup)
monitor = {
    "top": 50,      # Y-position of the top edge (absolute screen coordinate)
    "left": 0,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 1050  # Capture height
}

# Define the minimum confidence for YOLO detections to process
MIN_CONFIDENCE = 0.5

# Health Bar Configuration
# Adjust these parameters based on your health bar's appearance and position
health_bars = [
    {
        "name": "let_health",
        "top": 125,
        "left": 75,
        "width": 200,
        "height": 100,
         "color_lower": np.array([220, 30, 80]),    # Lower HSV bound for health bar color
    "color_upper": np.array([225, 40, 100]),    # Upper HSV bound for health bar colo
    "color2_lower": np.array([235, 180, 225]),    # Lower HSV bound for health bar color
    "color2_upper": np.array([255, 225, 255]),    # Upper HSV bound for health bar colo
    },
    {
        "name": "ret_health",
        "top": 125,
        "left": 450,
        "width": 200,
        "height": 100,
         "color_lower": np.array([220, 30, 80]),    # Lower HSV bound for health bar color
    "color_upper": np.array([225, 40, 100]),    # Upper HSV bound for health bar colo
    "color2_lower": np.array([235, 180, 225]),    # Lower HSV bound for health bar color
    "color2_upper": np.array([255, 225, 255]),    # Upper HSV bound for health bar colo
    },
    {
        "name": "lft_health",
        "top": 750,
        "left": 75,
        "width": 200,
        "height": 100,
         "color_lower": np.array([220, 30, 80]),    # Lower HSV bound for health bar color
    "color_upper": np.array([225, 40, 100]),    # Upper HSV bound for health bar colo
    "color2_lower": np.array([235, 180, 225]),    # Lower HSV bound for health bar color
    "color2_upper": np.array([255, 225, 255]),    # Upper HSV bound for health bar colo
    },
    {
        "name": "rft_health",
        "top": 750,
        "left": 450,
        "width": 200,
        "height": 100,
         "color_lower": np.array([220, 30, 80]),    # Lower HSV bound for health bar color
    "color_upper": np.array([225, 40, 100]),    # Upper HSV bound for health bar colo
    "color2_lower": np.array([235, 180, 225]),    # Lower HSV bound for health bar color
    "color2_upper": np.array([255, 225, 255]),    # Upper HSV bound for health bar colo
    },
    {
        "name": "cfk_health",
        "top": 950,
        "left": 275,
        "width": 200,
        "height": 100,
         "color_lower": np.array([220, 30, 80]),    # Lower HSV bound for health bar color
    "color_upper": np.array([225, 40, 100]),    # Upper HSV bound for health bar colo
    "color2_lower": np.array([235, 180, 225]),    # Lower HSV bound for health bar color
    "color2_upper": np.array([255, 225, 255]),    # Upper HSV bound for health bar colo
    },
    {
        "name": "cek_health",
        "top": 0,
        "left": 275,
        "width": 200,
        "height": 100,
         "color_lower": np.array([220, 30, 80]),    # Lower HSV bound for health bar color
    "color_upper": np.array([225, 40, 100]),    # Upper HSV bound for health bar colo
    "color2_lower": np.array([235, 180, 225]),    # Lower HSV bound for health bar color
    "color2_upper": np.array([255, 225, 255]),    # Upper HSV bound for health bar colo
    },
]

# ---------------------------
# Helper Functions
# ---------------------------

def calculate_health_percentage(frame, health_bar):
    """
    Calculates the health percentage based on the filled length of the health bar.
    
    :param frame: The captured frame from the screen.
    :param health_bar: Dictionary containing health bar configuration.
    :return: Tuple containing health percentage (0 to 100) and the mask image.
    """
    top = health_bar["top"]
    left = health_bar["left"]
    width = health_bar["width"]
    height = health_bar["height"]
    color_lower = health_bar["color_lower"]
    color_upper = health_bar["color_upper"]
    color2_lower = health_bar["color2_lower"]
    color2_upper = health_bar["color2_upper"]
    
    # Extract the ROI for the health bar
    roi = frame[top:top+height, left:left+width]
    
    # Check if ROI is valid
    if roi.size == 0:
        print(f"Error: ROI for {health_bar['name']} is empty. Check monitor coordinates.")
        return 0, None
    
    # Convert ROI to HSV color space for color detection
    hsv = roi
    
    # Create a mask for the health bar's color
    mask = cv2.inRange(hsv, color_lower, color_upper)

    
    
    # Apply morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    mask2 = cv2.inRange(hsv, color2_lower, color2_upper)
    mask2 = cv2.erode(mask2, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=2)
   
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour corresponds to the filled health bar
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Calculate the filled width
        max_w = 78
        filled_width = w
        # print(filled_width)
        # Calculate health percentage
        health_percentage = (filled_width / max_w) * 100
    else:
        
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if contours:
            print("found on secondary")
            # Assume the largest contour corresponds to the filled health bar
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Calculate the filled width
            max_w = 78
            filled_width = w
            # print(filled_width)
            # Calculate health percentage
            health_percentage = (filled_width / max_w) * 100
            # If no contours are found, assume 0% health
        else:
            health_percentage = 0
    
    return health_percentage, mask

def draw_health_bar(frame, health_bar, health_percentage):
    """
    Draws the health bar and its percentage on the frame.
    
    :param frame: The annotated frame to draw on.
    :param health_bar: Dictionary containing health bar configuration.
    :param health_percentage: Calculated health percentage.
    """
    top = health_bar["top"]
    left = health_bar["left"]
    width = health_bar["width"]
    height = health_bar["height"]
    
    # Draw the outer rectangle
    cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
    
    # Draw the filled portion based on health percentage
    filled_width = int((health_percentage / 100) * width)
    cv2.rectangle(frame, (left, top), (left + filled_width, top + height), (0, 255, 0), -1)  # Green filled bar
    
    # Display the health percentage
    cv2.putText(
        frame,
        f'{health_bar["name"]}: {health_percentage:.2f}%',
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

# ---------------------------
# Main Loop
# ---------------------------

frame_count = 0
start_time = time.time()

while True:
    try:
        frame_count += 1

        # Grab screen
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Run YOLO inference
        results = model(frame, size=640)  # AutoShape handles resizing

        # Render YOLO detections on the frame
        results.render()  # This modifies results.imgs in place
        annotated_frame = np.squeeze(results.render())  # Ensure it's a 2D image

        # Iterate over predefined health bars
        for health_bar in health_bars:
            health_percentage, mask = calculate_health_percentage(frame, health_bar)
            
            # Draw Health Bar on the frame
            draw_health_bar(annotated_frame, health_bar, health_percentage)
            
            # Log the health percentage
            logging.info(f'{health_bar["name"]}: {health_percentage:.2f}%')

            # Optional Debugging: Print to console
            print(f'{health_bar["name"]}: {health_percentage:.2f}%')

            # Optional Debugging: Display the mask
            # cv2.imshow(f"Mask - {health_bar['name']}", mask)
            # cv2.waitKey(1)

            # Optional Debugging: Save the mask image
            # cv2.imwrite(f"mask_{health_bar['name']}.png", mask)

        # Convert annotated_frame from RGB to BGR for OpenCV display
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Show the resulting frame
        cv2.imshow("YOLOv5 Detections with Health Monitoring", annotated_frame_bgr)

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

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")  # Optional: Print error to console for immediate feedback

# Cleanup
cv2.destroyAllWindows()
