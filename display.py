import numpy as np
import cv2
from mss import mss

# ---------------------------
# Configuration
# ---------------------------

# Initialize MSS screen capture
sct = mss()

# Define a bounding box for LDPlayer’s window (example coordinates, adjust to your setup)

monitor = {
    "top": 50,      # Y-position of the top edge (absolute screen coordinate)
    "left": 760,      # X-position of the left edge (absolute screen coordinate)
    "width": 720,   # Capture width
    "height": 1300  # Capture height
}

# ---------------------------
# Main Loop
# ---------------------------

while True:
    try:
        # Grab screen
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        
        # Check if the frame was captured correctly
        if frame.size == 0:
            print("Error: Captured frame is empty. Check monitor coordinates and ensure the target window is visible.")
            continue
        
        # Convert from BGRA to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Calculate Health Percentage
        # health_percentage, mask, mask2 = calculate_health_percentage(frame, health_bar)

        # Draw Health Bar on the frame
        # draw_health_bar(frame, health_bar, health_percentage)
        cv2.imshow("frame", frame)
        # Display the mask image (for debugging)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")  # Optional: Print error to console for immediate feedback

# Cleanup
cv2.destroyAllWindows()
