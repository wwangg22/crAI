import torch
import numpy as np

class YoloModel():
    def __init__(self, model_path, img_size, deck=False):
        self.model = torch.hub.load(
            './yolov5',  # Local path to YOLOv5 repository
            'custom',
            path=model_path,  # Path to your custom YOLOv5 model weights
            source='local'
        )
        self.img_size = img_size

        self.deck = deck

    def predict(self, frame, x_ranges=[(0.2,0.4), (0.4,0.6), (0.6,0.8),(0.8,1.0)], min_size=3600):
        results = self.model(frame, size=self.img_size)
        detections = results.xyxy[0]

        if self.deck:
            if x_ranges is None or len(x_ranges) != 4:
                # You could raise an error or supply a default set of ranges here:
                raise ValueError("x_ranges must be a list of 4 (minX, maxX) tuples when deck=True.")

            # Prepare structure to store the best detection (highest conf) in each of 4 slots
            slot_detections = [None, None, None, None]

            # Get frame width/height for normalization
            h, w, _ = frame.shape

            # Loop over all detections
            for *box, conf, cls in detections:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)

                # Only consider if area > min_size
                if area < min_size:
                    continue

                # Compute normalized x-center
                x_center = (x1 + x2) / 2.0
                x_center_norm = x_center / w

                # Find which slot this detection belongs to
                for i, (minX, maxX) in enumerate(x_ranges):
                    if minX <= x_center_norm <= maxX:
                        # Update if we don't have a detection yet or if this one is higher confidence
                        if slot_detections[i] is None or conf > slot_detections[i]['conf']:
                            slot_detections[i] = {
                                'class_id': int(cls),  # YOLO class ID as integer
                                'conf': float(conf)
                            }
                        break

            # Build final 16-element output array
            #  -> 4 slots, each = [valid_bit, bit1, bit2, bit3]
            #  -> total length = 4 * 4 = 16
            output_array = np.zeros(16, dtype=np.int32)

            for slot_idx in range(4):
                det = slot_detections[slot_idx]
                # If there's a detection in that slot
                if det is not None:
                    card_id = det['class_id']
                    # Set valid bit to 1
                    output_array[4 * slot_idx] = 1
                    # Convert class_id (0..7) to 3-bit binary
                    # e.g. class_id=5 -> '101'
                    bits = f"{card_id:03b}"  # zero-padded to length=3
                    # Place the 3 bits after the valid bit
                    output_array[4 * slot_idx + 1] = int(bits[0])
                    output_array[4 * slot_idx + 2] = int(bits[1])
                    output_array[4 * slot_idx + 3] = int(bits[2])
                else:
                    # No detection found => [0, 0, 0, 0] stays as default
                    pass

            return output_array
         # --------------------------------------------------
        # 2) Else scenario: "troops on a board"
        # --------------------------------------------------
        else:
            """
            We have a maximum of 10 troops.
            Each troop has 6 info points:
              1) existence bit (1=troop, 0=none/padding)
              2) normalized x-center
              3) normalized y-center
              4) class bit 1
              5) class bit 2
              6) class bit 3

            => 10 troops * 6 info points = 60 total array elements.
            If fewer than 10 troops, we pad with zeros.
            If more than 10, keep only the top 10 by confidence.
            """
            h, w, _ = frame.shape

            # Sort detections by confidence descending
            if len(detections) > 0:
                # detections[:,4] is confidence
                _, indices = torch.sort(detections[:, 4], descending=True)
                detections = detections[indices]

            # Prepare 60-element output array
            # Troop i occupies indices [i*6 .. i*6+5]
            output_array = np.zeros(60, dtype=np.float32)

            troop_count = min(len(detections), 10)
            for i in range(troop_count):
                x1, y1, x2, y2, conf, cls = detections[i]
                # compute center
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0

                base_idx = i * 6
                # 1) Existence bit
                output_array[base_idx + 0] = 1  
                # 2) normalized x-center
                output_array[base_idx + 1] = x_center / w  
                # 3) normalized y-center
                output_array[base_idx + 2] = y_center / h  

                # 4,5,6) Class bits (3 bits for class ID in [0..7])
                class_id = int(cls)
                bits = f"{class_id:03b}"  # zero-padded to length=3
                output_array[base_idx + 3] = int(bits[0])
                output_array[base_idx + 4] = int(bits[1])
                output_array[base_idx + 5] = int(bits[2])

            return output_array


