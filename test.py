import cv2
import numpy as np
import time

def sift_homography_match(scene_gray, template_gray, min_match_count=10, ratio_thresh=0.7):
    """
    Perform SIFT feature matching with homography to find the template in the scene.
    Returns a list of bounding boxes [x_min, y_min, x_max, y_max]
    if enough matches are found. Otherwise returns an empty list.
    """
    # Create SIFT (requires opencv-contrib-python in most cases)
    sift = cv2.SIFT_create()

    # Detect and compute descriptors
    kp_template, des_template = sift.detectAndCompute(template_gray, None)
    kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)

    # If no descriptors found, return immediately
    if des_template is None or des_scene is None:
        return []

    # FLANN-based matching parameters for SIFT
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN match, k=2
    matches = flann.knnMatch(des_template, des_scene, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Check if enough "good" matches are found
    if len(good_matches) < min_match_count:
        return []

    # Extract matched keypoints
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return []

    # Warp the corners of the template image to get bounding box in the scene
    h, w = template_gray.shape
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # Convert to a bounding box
    x_coords = transformed_corners[:,0,0]
    y_coords = transformed_corners[:,0,1]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    return [(x_min, y_min, x_max, y_max)]

def draw_sift_result(scene_rgb, boxes, label,
                     box_color=(0, 255, 0), text_color=(255, 255, 255)):
    """
    Draws the bounding boxes from sift_homography_match on the scene with a label.
    boxes is a list of [x_min, y_min, x_max, y_max] or empty if no match.
    """
    for (x_min, y_min, x_max, y_max) in boxes:
        # Draw rectangle
        cv2.rectangle(scene_rgb, (x_min, y_min), (x_max, y_max), box_color, 2)
        # Put label above the rectangle
        cv2.putText(scene_rgb, label, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
    return scene_rgb

def main():
    scene_filename = "clashroyalesc1.png"
    template_files = [
        "enemy_hog_rider_backward.png",
        "enemy_hog_rider_forward.png",
        "friend_hog_rider_backward.png",
        "friend_hog_rider_forward.png"
    ]

    # Load scene
    scene_rgb = cv2.imread(scene_filename)
    if scene_rgb is None:
        print(f"Error loading scene image: {scene_filename}")
        return
    scene_gray = cv2.cvtColor(scene_rgb, cv2.COLOR_BGR2GRAY)

    # We'll draw all bounding boxes on one copy of the scene
    scene_sift_result = scene_rgb.copy()

    # Time the entire SIFT process
    t0 = time.time()

    for tfile in template_files:
        template_rgb = cv2.imread(tfile)
        if template_rgb is None:
            print(f"Error loading template image: {tfile}")
            continue

        template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

        # Perform SIFT matching
        boxes = sift_homography_match(scene_gray, template_gray,
                                      min_match_count=10, ratio_thresh=0.7)

        # Draw results if we found any
        if len(boxes) > 0:
            scene_sift_result = draw_sift_result(scene_sift_result, boxes, label=tfile)
        else:
            print(f"No matches found for: {tfile}")

    t1 = time.time()
    print(f"SIFT matching done in {t1 - t0:.4f} seconds")

    # Save result
    out_filename = "final_sift_result.png"
    cv2.imwrite(out_filename, scene_sift_result)
    print(f"Result saved as {out_filename}")

    # If you want to show the result in a window (with a GUI):
    # cv2.imshow("SIFT Result", scene_sift_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
