#get video with id in list
import os
import shutil

# Danh sách các ID cần lọc
list_ids = [17, 21, 26, 33, 39, 44, 50, 51, 56, 64]  # Thay thế bằng danh sách ID của bạn

# Đường dẫn đến thư mục chứa video
source_folder = r'D:\yolo_data\sign language new\all'  # Thay thế bằng đường dẫn thực tế

# Đường dẫn đến thư mục mà bạn muốn lưu các video đã lọc
destination_folder = r'data\raw_data'  # Thay thế bằng đường dẫn thực tế

# Tạo thư mục đích nếu nó chưa tồn tại
os.makedirs(destination_folder, exist_ok=True)

# Duyệt qua tất cả các file trong thư mục nguồn
for filename in os.listdir(source_folder):
    if filename.endswith(".mp4"):
        # Tách ID từ tên file (giả sử ID là phần tử đầu tiên trong tên file, ngăn cách bởi dấu '_')
        video_id = int(filename.split('_')[0])

        # Kiểm tra nếu ID của video nằm trong danh sách ID cần lọc
        if video_id in list_ids:
            # Đường dẫn đầy đủ của file nguồn và đích
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Sao chép file
            shutil.copy2(source_file, destination_file)
