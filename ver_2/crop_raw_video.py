import cv2
import mediapipe as mp
import os
import concurrent.futures
import os
import glob
from tqdm import tqdm

def crop_video(input_path, output_path):
    # Khởi tạo MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Đường dẫn đến video gốc và thư mục lưu video mới
    input_video_path = input_path   

    # Đọc video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo writer cho video mới
    output_video_path = output_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width // 2, height))

    # Xử lý video
    first_frame = True
    nose_point = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xác định điểm mũi trong khung hình đầu tiên
        if first_frame:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    nose_point = face_landmarks.landmark[0]  # ID 0 là điểm mũi
                    break
            first_frame = False

            # Tính toán vùng crop
            if nose_point:
                nose_x = int(nose_point.x * width)
                x_start = max(nose_x - width // 4, 0)
                x_end = min(nose_x + width // 4, width)

        # Crop và lưu khung hình
        if nose_point:
            cropped_frame = frame[:, x_start:x_end]
            out.write(cropped_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video(file_path):
    base_name = os.path.basename(file_path)
    new_name = base_name.replace(".mp4", "_cropped.mp4")
    output_path = os.path.join("crop_video", new_name)
    crop_video(file_path, output_path)

def main():
    input_folder = r'data\raw_data'  # Thay thế bằng đường dẫn thư mục chứa video
    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))

    os.makedirs('crop_video', exist_ok=True)

    # Tạo 6 luồng để chạy đa luồng
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Sử dụng tqdm để tạo thanh tiến trình
        futures = [executor.submit(process_video, file) for file in video_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_files)):
            future.result()

if __name__ == "__main__":
    main()