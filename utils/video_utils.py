import cv2
def video_read(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def video_write(output_frames, video_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    h, w, _ = output_frames[0].shape
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for frame in output_frames:
        out.write(frame)
    out.release()