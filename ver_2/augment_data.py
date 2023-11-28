import cv2
import os
import numpy as np
import random

def rotate_video(video_path, output_folder, angle):
    # Đọc video gốc
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số cơ bản của video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo đối tượng lưu video
    base_name = os.path.basename(video_path)
    method = 'left' if angle < 0 else 'right'
    new_name = os.path.splitext(base_name)[0] + f"_rotated_{method}.mp4"
    output_path = os.path.join(output_folder, new_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Xoay từng frame và lưu vào video mới
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tạo ma trận xoay
        M = cv2.getRotationMatrix2D((frame_width/2, frame_height/2), angle, 1)
        rotated_frame = cv2.warpAffine(frame, M, (frame_width, frame_height))

        # Ghi frame đã xoay vào file đầu ra
        out.write(rotated_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()

# Hàm xoay video sang trái 5 độ
def rotate_video_left(video_path, output_folder):
    rotate_video(video_path, output_folder, -5)

# Hàm xoay video sang phải 5 độ
def rotate_video_right(video_path, output_folder):
    rotate_video(video_path, output_folder, 5)

def shrink_video(video_path, output_folder):
    # Đọc video gốc
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số cơ bản của video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sinh tỷ lệ thu nhỏ ngẫu nhiên
    scale = random.uniform(0.75, 0.9)

    # Tính kích thước mới cho frame thu nhỏ
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)

    # Tạo đối tượng lưu video
    base_name = os.path.basename(video_path)
    new_name = os.path.splitext(base_name)[0] + "_shrinked.mp4"
    output_path = os.path.join(output_folder, new_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Xử lý và lưu từng frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Áp dụng thu nhỏ cho frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Tạo một khung hình mới với kích thước gốc và màu đen
        new_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

        # Tính toán vị trí để đặt frame thu nhỏ
        x_offset = (frame_width - new_width) // 2
        y_offset = (frame_height - new_height) // 2

        # Đặt frame thu nhỏ vào giữa khung hình mới
        new_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

        # Ghi frame mới vào file đầu ra
        out.write(new_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()

def shrink_video_linear(video_path, output_folder):
    # Đọc video gốc
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số cơ bản của video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo đối tượng lưu video
    base_name = os.path.basename(video_path)
    new_name = os.path.splitext(base_name)[0] + "_shrinking_linearly.mp4"
    output_path = os.path.join(output_folder, new_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Xử lý và lưu từng frame
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Tính toán tỷ lệ thu nhỏ cho frame hiện tại
        scale = 1 - (0.3 * frame_num / total_frames) # từ 1x xuống 0.7x

        # Tính kích thước mới cho frame thu nhỏ
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Áp dụng thu nhỏ cho frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Tạo một khung hình mới với kích thước gốc và màu đen
        new_frame = np.zeros((frame_height, frame_width, 3), np.uint8)

        # Tính toán vị trí để đặt frame thu nhỏ
        x_offset = (frame_width - new_width) // 2
        y_offset = (frame_height - new_height) // 2

        # Đặt frame thu nhỏ vào giữa khung hình mới
        new_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

        # Ghi frame mới vào file đầu ra
        out.write(new_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()

def zoom_video_fixed(video_path, output_folder):
    # Đọc video gốc
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số cơ bản của video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sinh tỷ lệ zoom ngẫu nhiên từ 1.05 đến 1.15
    scale = random.uniform(1.05, 1.15)

    # Tạo đối tượng lưu video
    base_name = os.path.basename(video_path)
    new_name = os.path.splitext(base_name)[0] + "_zoom_fixed.mp4"
    output_path = os.path.join(output_folder, new_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Xử lý và lưu từng frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Áp dụng zoom cho frame
        zoomed_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cropped_frame = zoomed_frame[int((zoomed_frame.shape[0] - frame_height) / 2):int((zoomed_frame.shape[0] + frame_height) / 2), int((zoomed_frame.shape[1] - frame_width) / 2):int((zoomed_frame.shape[1] + frame_width) / 2)]

        # Ghi frame đã zoom vào file đầu ra
        out.write(cropped_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()

def zoom_video_linear(video_path, output_folder):
    # Đọc video gốc
    cap = cv2.VideoCapture(video_path)

    # Lấy thông số cơ bản và tổng số frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo đối tượng lưu video
    base_name = os.path.basename(video_path)
    new_name = os.path.splitext(base_name)[0] + "_zoom_linear.mp4"
    output_path = os.path.join(output_folder, new_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Xử lý và lưu từng frame
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Tính toán tỷ lệ zoom cho frame hiện tại
        scale = 1 + 0.15 * frame_num / total_frames # từ 1x đến 1.15x

        # Áp dụng zoom cho frame
        zoomed_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cropped_frame = zoomed_frame[int((zoomed_frame.shape[0] - frame_height) / 2):int((zoomed_frame.shape[0] + frame_height) / 2), int((zoomed_frame.shape[1] - frame_width) / 2):int((zoomed_frame.shape[1] + frame_width) / 2)]

        # Ghi frame đã zoom vào file đầu ra
        out.write(cropped_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()

import concurrent.futures
import os
from tqdm import tqdm

# Định nghĩa hàm để xử lý một video với tất cả các phương pháp
def process_video(video_path, output_folder):
    rotate_video_left(video_path, output_folder)
    rotate_video_right(video_path, output_folder)
    shrink_video(video_path, output_folder)
    shrink_video_linear(video_path, output_folder)
    zoom_video_fixed(video_path, output_folder)
    zoom_video_linear(video_path, output_folder)

def process_all_videos(input_folder, output_folder):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lấy danh sách tất cả các video trong thư mục đầu vào
    videos = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # Sử dụng ThreadPoolExecutor để xử lý đa luồng
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Áp dụng hàm xử lý cho từng video và tạo một danh sách futures
        futures = {executor.submit(process_video, video, output_folder): video for video in videos}

        # Sử dụng tqdm để theo dõi tiến trình
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(videos), desc="Processing Videos"):
            future.result()

# Gọi hàm chính
input_folder = 'crop_video'
output_folder = 'augmented_data'
process_all_videos(input_folder, output_folder)