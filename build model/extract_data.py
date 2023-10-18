import cv2
import mediapipe as mp
import numpy as np
import os
import csv

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

def create_data(video_path):
    cap = cv2.VideoCapture(video_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    LEFT_HAND_LANDMARKS = [15, 17, 19, 21]
    RIGHT_HAND_LANDMARKS = [16, 18, 20, 22]

    left_hand_coordinates = []
    right_hand_coordinates = []
    recording = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

                if left_hand_raised or right_hand_raised:
                    for landmark in LEFT_HAND_LANDMARKS:
                        point = mp_drawing._normalized_to_pixel_coordinates(closest_person.landmark[landmark].x,
                                                                            closest_person.landmark[landmark].y,
                                                                            frame.shape[1], frame.shape[0])
                        if point:
                            cv2.circle(frame, (point[0] + offset_x, point[1] + offset_y), 5, (0, 255, 0), -1)

                    for landmark in RIGHT_HAND_LANDMARKS:
                        point = mp_drawing._normalized_to_pixel_coordinates(closest_person.landmark[landmark].x,
                                                                            closest_person.landmark[landmark].y,
                                                                            frame.shape[1], frame.shape[0])
                        if point:
                            cv2.circle(frame, (point[0] + offset_x, point[1] + offset_y), 5, (128, 0, 128), -1)

                    if chest_point:
                        cv2.circle(frame, (chest_point[0] + offset_x, chest_point[1] + offset_y), 10, (0, 255, 0), -1)

                if not recording and (left_hand_raised or right_hand_raised):
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
            if not os.path.exists("extracted_data"):
                os.makedirs("extracted_data")

            # Create a blank white image
            white_image = 255 * np.ones((frame.shape[0], frame.shape[1], 3), np.uint8)

            # Draw the hand coordinates onto the white image
            for coord in left_hand_coordinates:
                cv2.circle(white_image, coord, 5, (0, 255, 0), -1)
            for coord in right_hand_coordinates:
                cv2.circle(white_image, coord, 5, (128, 0, 128), -1)

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


            output_filename_image = os.path.join("extracted_data", os.path.basename(video_path).replace(".mp4", ".png"))
            cv2.imwrite(output_filename_image, white_image)

            # Save the hand coordinates to a CSV file
            output_filename_csv = os.path.join("extracted_data", os.path.basename(video_path).replace(".mp4", ".csv"))
            with open(output_filename_csv, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Hand', 'X', 'Y'])
                for coord in left_hand_coordinates:
                    csv_writer.writerow(['Left', coord[0], coord[1]])
                for coord in right_hand_coordinates:
                    csv_writer.writerow(['Right', coord[0], coord[1]])

        # # cv2.imshow("Pose Keypoints", frame)
        # # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # dataset folder path
    folder_path = 'all'

    # List all videos
    video_paths = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            if file.endswith('.avi') or file.endswith('.mp4'):
                video_paths.append(os.path.join(dirpath, file))


    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def process_video(video):
        create_data(video)
        return video 

    #multi-threading
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(process_video, video_paths), total=len(video_paths)))
