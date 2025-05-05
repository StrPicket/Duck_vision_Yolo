import cv2
import numpy as np

# === CONFIGURATION ===
video_path = "assets/DuckVideoSampled.mp4"
real_distance_m = 0.3  # Real-world distance between the two clicked points
intrinsics_file = "camera_information/intrinsics.npy"
essential_matrices_file = "camera_information/essential_matrices.npy"

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")

def get_calibration_points(frame):
    global clicked_points
    clicked_points = []

    cv2.imshow("Click TWO calibration points", frame)
    cv2.setMouseCallback("Click TWO calibration points", mouse_callback)

    while len(clicked_points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return clicked_points[0], clicked_points[1]

def estimate_intrinsics_from_two_points(p1, p2, real_distance, frame_shape):
    h, w = frame_shape
    cx, cy = w // 2, h // 2

    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    approx_focal = pixel_distance / real_distance

    K = np.array([
        [approx_focal, 0, cx],
        [0, approx_focal, cy],
        [0, 0, 1]
    ])
    return K

def compute_essential_matrix(img1, img2, K):
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E

def process_video(video_path, real_distance):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading first frame.")
        return

    point1, point2 = get_calibration_points(first_frame)
    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    K = estimate_intrinsics_from_two_points(point1, point2, real_distance, gray_prev.shape)
    print("\nEstimated Intrinsic Matrix K:\n", K)

    # Save intrinsics
    np.save(intrinsics_file, K)
    print(f"\nSaved intrinsics to '{intrinsics_file}'")

    essential_matrices = []
    frame_idx = 1

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        E = compute_essential_matrix(gray_prev, gray_curr, K)

        if E is not None:
            essential_matrices.append(E)
            print(f"Frame {frame_idx}: Essential matrix computed.")
        else:
            print(f"Frame {frame_idx}: Essential matrix not computed.")

        gray_prev = gray_curr
        frame_idx += 1

    cap.release()
    np.save(essential_matrices_file, np.array(essential_matrices))
    print(f"\nSaved {len(essential_matrices)} essential matrices to '{essential_matrices_file}'.")

# === Run It ===
process_video(video_path, real_distance_m)
