import cv2

def extract_and_save_every_10th_frame(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    # Get original video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'avc1' or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps // 10, (width, height))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            out.write(frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved {saved_count} frames to video '{output_path}'.")

# Example usage
input_video = "assets/DuckVideo.mp4"
output_video = "assets/DuckVideoSampled.mp4"
extract_and_save_every_10th_frame(input_video, output_video)
