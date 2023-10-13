import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

def find_closest_person_to_center(pose_landmarks, frame_width, frame_height):
    min_distance = float('inf')
    closest_person = None
    chest_point = None
    for landmarks in pose_landmarks:
        chest_x = (landmarks.landmark[11].x + landmarks.landmark[12].x) / 2 * frame_width
        chest_y = (landmarks.landmark[11].y + landmarks.landmark[12].y) / 2 * frame_height
        distance_to_center = abs(frame_width/2 - chest_x)
        if distance_to_center < min_distance:
            min_distance = distance_to_center
            closest_person = landmarks
            chest_point = (int(chest_x), int(chest_y))
    return closest_person, chest_point

def hand_raised_above_hip(pose_landmarks, landmark_idx, adjusted_hip_y):
    hand_y = pose_landmarks.landmark[landmark_idx].y
    return hand_y < adjusted_hip_y

def adjust_hip_position(hip_y, chest_y):
    return (hip_y + chest_y) / 2

def face_drawing(frame, nose_point, radius=50, color=(255, 0, 0), thickness=2):
    if nose_point:
        cv2.circle(frame, nose_point, radius, color, thickness)

def inference(video_path, model_path):
    cap = cv2.VideoCapture(video_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    LEFT_HAND_LANDMARKS = [15, 17, 19, 21]
    RIGHT_HAND_LANDMARKS = [16, 18, 20, 22]

    left_hand_coordinates = []
    right_hand_coordinates = []
    recording = False
    infer = False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    display_duration = 10000000  # max seconds
    frames_to_display = fps * display_duration
    count = 0
    infer_count = 0
    infer_duration = 3 * fps
    fps_count = 0
    current_fps = fps

    current_prediction = None  

    #load_model
    model = load_model(model_path)

    while cap.isOpened():
        start_time = time.time()  # Start time for FPS calculation
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            nose_point = mp_drawing._normalized_to_pixel_coordinates(
                results.pose_landmarks.landmark[0].x,
                results.pose_landmarks.landmark[0].y,
                frame.shape[1], frame.shape[0]
            )

            if nose_point:
                offset_x = frame.shape[1] // 2 - nose_point[0]
                offset_y = frame.shape[0] // 4 - nose_point[1]

                for landmark in results.pose_landmarks.landmark:
                    point = mp_drawing._normalized_to_pixel_coordinates(
                        landmark.x, landmark.y, frame.shape[1], frame.shape[0]
                    )

                    if point:
                        adjusted_point = (point[0] + offset_x, point[1] + offset_y)
                        cv2.circle(frame, adjusted_point, 5, (0, 255, 0), -1)

            closest_person, chest_point = find_closest_person_to_center([results.pose_landmarks], frame.shape[1], frame.shape[0])
            if closest_person:
                adjusted_left_hip_y = adjust_hip_position(closest_person.landmark[23].y, closest_person.landmark[11].y)
                adjusted_right_hip_y = adjust_hip_position(closest_person.landmark[24].y, closest_person.landmark[12].y)

                left_hand_raised = hand_raised_above_hip(closest_person, 15, adjusted_left_hip_y)
                right_hand_raised = hand_raised_above_hip(closest_person, 16, adjusted_right_hip_y)

                if not recording and (left_hand_raised or right_hand_raised):
                    infer = False
                    recording = True

                if recording:
                    for landmark in LEFT_HAND_LANDMARKS:
                        point = mp_drawing._normalized_to_pixel_coordinates(closest_person.landmark[landmark].x,
                                                                            closest_person.landmark[landmark].y,
                                                                            frame.shape[1], frame.shape[0])
                        if point:
                            left_hand_coordinates.append((point[0] + offset_x, point[1] + offset_y))

                    for landmark in RIGHT_HAND_LANDMARKS:
                        point = mp_drawing._normalized_to_pixel_coordinates(closest_person.landmark[landmark].x,
                                                                            closest_person.landmark[landmark].y,
                                                                            frame.shape[1], frame.shape[0])
                        if point:
                            right_hand_coordinates.append((point[0] + offset_x, point[1] + offset_y))

        if recording:

            # Create a blank white image
            white_image = 255 * np.ones((frame.shape[0], frame.shape[1], 3), np.uint8)

            # left_color_lst = [(0,238,0), (0,238,238), (188,210,238), (238,213,210)]
            # right_color_lst = [(145,44,238), (220,20,60), (255,52,179), (255,127,0)]

            left_color_lst = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]
            right_color_lst = [(128, 0, 128), (128, 0, 128), (128, 0, 128), (128, 0, 128)]

            # Đảm bảo rằng có ít nhất một phần tử trong danh sách màu
            if left_color_lst and right_color_lst:
                for i, coord in enumerate(left_hand_coordinates):
                    color = left_color_lst[i % len(left_color_lst)]  # Sử dụng toán tử % để lặp lại danh sách màu
                    cv2.circle(white_image, coord, 5, color, -1)
                
                for i, coord in enumerate(right_hand_coordinates):
                    color = right_color_lst[i % len(right_color_lst)]  # Sử dụng toán tử % để lặp lại danh sách màu
                    cv2.circle(white_image, coord, 5, color, -1)


            # Draw a smiling face at 1/4 height and 1/2 width of the white image
            center_point = (frame.shape[1] // 2, frame.shape[0] // 4)
            radius = 150  # Adjust the radius as needed

            # Draw the face circle
            cv2.circle(white_image, center_point, radius, (0, 0, 255), 3)  # Outline circle

            # Draw the eyes
            eye_radius = 5  # Adjust the eye radius as needed
            left_eye_point = (center_point[0] - radius//2, center_point[1] - radius//2)
            right_eye_point = (center_point[0] + radius//2, center_point[1] - radius//2)
            cv2.circle(white_image, left_eye_point, eye_radius, (0, 0, 255), 3)
            cv2.circle(white_image, right_eye_point, eye_radius, (0, 0, 255), 3)

            # Draw the mouth
            start_angle = 0
            end_angle = 180
            axes = (radius//2, radius//4)
            mouth_point = (center_point[0], center_point[1] + radius//3)
            cv2.ellipse(white_image, mouth_point, axes, 0, start_angle, end_angle, (0, 0, 255), 3)


            if count < frames_to_display:
                white_image_resized = cv2.resize(white_image, (w//3, h//3))
                x_offset = w - white_image_resized.shape[1]
                y_offset = h - white_image_resized.shape[0]
                frame[y_offset:h, x_offset:w] = white_image_resized
                count += 1

            # Clear the lists for the next recording
            if not (left_hand_raised or right_hand_raised):
                recording = False
                infer = True
                left_hand_coordinates.clear()
                right_hand_coordinates.clear()

        if infer == True:
            if infer_count < infer_duration:
                if infer_count == 0:
                    #inference
                    # Resize the image to 256x256
                    img = cv2.resize(white_image, (256, 256))
                    img = np.expand_dims(img, axis=0)  # Add an extra dimension for batch size
                    #predict
                    predictions = model.predict(img)
                    predicted_labels = np.argmax(predictions, axis=1)
                    # Cập nhật kết quả dự đoán hiện tại
                    current_prediction = f"Predict label: {predicted_labels[0]}"  # Modify this line as per your label mapping

                 # Add prediction text to the frame
                prediction_text = current_prediction  # Modify this line as per your label mapping
                
                # Get the size of the text
                text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

                # Calculate the center position of the text
                text_x = (frame.shape[1] - text_size[0] - 50)
                text_y = frame.shape[0] * 4 // 5  # Set the height to be at 1/5 of the frame height

                cv2.putText(frame, prediction_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                infer_count += 1
            else:
                infer = False
                infer_count = 0
                current_prediction = None  # Clear the current prediction when not inferring
            
        if fps_count == 200:
            # Calculate FPS
            end_time = time.time()
            current_fps = 1 / (end_time - start_time)
            fps_count = 0
        fps_count += 1
        fps = current_fps
        fps_text = f"FPS: {fps:.2f}"

        # Display FPS on the frame
        cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        frame = cv2.resize(frame, (w//2, h//2))

        cv2.imshow("Pose Keypoints", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
inference(r"KHKT\all.mp4", "models/sign_language_model.h5")