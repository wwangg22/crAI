import numpy as np
import cv2
from mss import mss
import logging
import os

MAX_GAP = 5          # Maximum gap between digits to group them into a number
MAX_Y_GAP = 5
MATCH_METHOD = cv2.TM_CCOEFF_NORMED
NMS_THRESHOLD = 0.5  # Non-Maximum Suppression threshold
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
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred


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


class TowerHealth():
    def __init__(self, level = 9):
        self.princess_hp_list = [1400, 1512, 1624, 1750, 1890, 2030, 2184, 2352, 2534, 2786, 3052, 3346, 3668]
        self.king_hp_list = [ 2400, 2568, 2736, 2904, 3096, 3312, 3528, 3768, 4008, 4392, 4824, 5304, 5832]
        self.max_princess_tower_hp = self.princess_hp_list[level-1]
        self.max_king_tower_hp = self.king_hp_list[level-1]
        self.friendly_princess_tower_hp = None
        self.friendly_king_tower_hp = None
        self.enemy_princess_tower_hp = None
        self.enemy_king_tower_hp = None
        self.level = level
        self.reward = 0

    def setLevel(self):
        self.max_princess_tower_hp = self.princess_hp_list[self.level-1]
        self.max_king_tower_hp = self.king_hp_list[self.level-1]

    def get_friendly_princess_tower_hp(self):
        if self.friendly_princess_tower_hp is not None:
            return self.friendly_princess_tower_hp / self.max_princess_tower_hp
    def get_friendly_king_tower_hp(self):
        if self.friendly_king_tower_hp is not None:
            return self.friendly_king_tower_hp / self.max_king_tower_hp

    def get_enemy_princess_tower_hp(self):
        if self.enemy_princess_tower_hp is not None:
            return self.enemy_princess_tower_hp / self.max_princess_tower_hp

    def get_enemy_king_tower_hp(self):
        if self.enemy_king_tower_hp is not None:
            return self.enemy_king_tower_hp / self.max_king_tower_hp
    
    def update(self, frame, tower_monitor, template_dir = "./num"):
        """ tower monitor:
        [
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
        """
        digit_templates = load_digit_templates(template_dir)
        for tower in tower_monitor:
            top = tower["top"]
            left = tower["left"]
            width = tower["width"]
            height = tower["height"]

            # Extract the ROI for the health bar from the pure frame
            roi = frame[top:top + height, left:left + width]

            # Check if ROI is valid
            if roi.size == 0:
                logging.error(f"ROI for {tower['name']} is empty. Check monitor coordinates.")
                print(f"Error: ROI for {tower['name']} is empty. Check monitor coordinates.")
                continue

            # Preprocess the ROI
            preprocessed_roi = preprocess_image(roi)

            # Perform template matching
            detections = perform_template_matching(preprocessed_roi, digit_templates)
            detections = filter_detections(detections)
            filtered_detections = non_max_suppression(detections)

            # Adjust positions relative to the entire frame
            for det in filtered_detections:
                det['position'] = (det['position'][0] + left, det['position'][1] + top)

            # Group detected digits into numbers
            strs = group_digits(filtered_detections)
            recognized_numbers = [int(x) for x in strs]


            if tower["name"][2] == 'k':
                if tower["name"][1] == "f":
                    if self.friendly_princess_tower_hp is None:
                        if len(recognized_numbers) > 0:
                            if self.level != int(max(recognized_numbers)):
                                self.level = int(max(recognized_numbers))
                                self.setLevel()
                            self.friendly_king_tower_hp = self.max_king_tower_hp
                            self.enemy_king_tower_hp = self.max_king_tower_hp
                        
                        continue
                    if self.friendly_princess_tower_hp is not None and \
                    self.friendly_princess_tower_hp[0] > 0 and \
                    self.friendly_princess_tower_hp[1] > 0:
                        continue
                    else:
                        if len(recognized_numbers) > 0:
                            if self.level in recognized_numbers:
                                ind = recognized_numbers.index(self.level)
                                recognized_numbers.pop(ind)
                            if len(recognized_numbers) > 0:
                                health = int(max(recognized_numbers))
                                if health > self.friendly_king_tower_hp - 900 and health < self.friendly_king_tower_hp:
                                    self.reward += (health - self.friendly_king_tower_hp)
                                    self.friendly_king_tower_hp = health
                else:
                    if self.friendly_princess_tower_hp is None:
                        if len(recognized_numbers) > 0:
                            if self.level != int(max(recognized_numbers)):
                                self.level = int(max(recognized_numbers))
                                self.setLevel()
                            self.friendly_king_tower_hp = self.max_king_tower_hp
                            self.enemy_king_tower_hp = self.max_king_tower_hp
                        continue
                    if self.enemy_princess_tower_hp is not None and \
                    self.enemy_princess_tower_hp[0] > 0 and \
                    self.enemy_princess_tower_hp[1] > 0:
                        continue
                    else:
                        if len(recognized_numbers) > 0:
                            if self.level in recognized_numbers:
                                ind = recognized_numbers.index(self.level)
                                recognized_numbers.pop(ind)
                            if len(recognized_numbers) > 0:
                                health = int(max(recognized_numbers))
                                print(recognized_numbers, health, self.enemy_king_tower_hp, self.max_king_tower_hp)
                                if health > self.enemy_king_tower_hp - 900 and health < self.enemy_king_tower_hp:
                                    self.reward += (self.enemy_king_tower_hp - health)
                                    self.enemy_king_tower_hp = health
            else:
                if tower["name"][1] == "f":
                    if self.friendly_princess_tower_hp is None:
                        self.friendly_princess_tower_hp = np.full((2,), self.max_princess_tower_hp)
                        continue

                    if len(recognized_numbers) > 0:
                        health = int(max(recognized_numbers))
                    else:
                        health = 0

                    if tower["name"][0] == "l":
                        # print(health,self.friendly_princess_tower_hp[0] - 900, self.max_princess_tower_hp,self.friendly_princess_tower_hp)
                        if health > self.friendly_princess_tower_hp[0] - 900 and health < self.friendly_princess_tower_hp[0]:
                            self.reward += (health - self.friendly_princess_tower_hp[0]) 
                            self.friendly_princess_tower_hp[0] = health
                    else:
                        if health > self.friendly_princess_tower_hp[1] - 900 and health < self.friendly_princess_tower_hp[1]:
                            self.reward += (health - self.friendly_princess_tower_hp[1])
                            self.friendly_princess_tower_hp[1] = health
                else:
                    if self.enemy_princess_tower_hp is None:
                        self.enemy_princess_tower_hp = np.full((2,), self.max_princess_tower_hp)
                        continue

                    if len(recognized_numbers) > 0:
                        health = int(max(recognized_numbers))
                    else:
                        health = 0

                    if tower["name"][0] == "l":
                        if health > self.enemy_princess_tower_hp[0] - 900 and health < self.enemy_princess_tower_hp[0]:
                            self.reward += (self.enemy_princess_tower_hp[0] - health)
                            self.enemy_princess_tower_hp[0] = health
                    else:
                        if health > self.enemy_princess_tower_hp[1] - 900 and health < self.enemy_princess_tower_hp[1]:
                            self.reward += (self.enemy_princess_tower_hp[1] - health)
                            self.enemy_princess_tower_hp[1] = health
    def reset(self):
        self.enemy_princess_tower_hp = None
        self.friendly_king_tower_hp = None
        self.friendly_princess_tower_hp = None
        self.enemy_king_tower_hp = None

    def getReward(self):
        reward = self.reward / 100
        self.reward = 0
        return reward

    def getAllHealth(self):
        if self.friendly_princess_tower_hp is not None and self.enemy_princess_tower_hp is not None and self.friendly_king_tower_hp is not None and self.enemy_king_tower_hp is not None:
            final_arr = np.concatenate((self.friendly_princess_tower_hp/self.max_princess_tower_hp, self.enemy_princess_tower_hp/self.max_princess_tower_hp, [self.friendly_king_tower_hp/self.max_king_tower_hp], [self.enemy_king_tower_hp/self.max_king_tower_hp]))
            return final_arr
        else:
            return np.zeros((6,))
        
    def getDim(self):
        return 6