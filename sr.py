import cv2
import numpy as np
import mss

def main():
    # Create an MSS instance
    with mss.mss() as sct:
        # Define which part of the screen to capture.
        # Adjust these values as needed.
        # You can also capture an entire monitor by specifying sct.monitors[1].
        monitor = {
            "top": 100,     # Y-position of the top edge
            "left": 0,    # X-position of the left edge
            "width": 732,   # Capture width
            "height": 952   # Capture height
        }
        
        while True:
            # Grab the data
            screenshot = sct.grab(monitor)
            
            # Convert the raw pixels into a NumPy array
            frame = np.array(screenshot)
            
            # The image from MSS is in BGRA format; convert to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Display the captured frame
            cv2.imshow("Screen Capture", frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
