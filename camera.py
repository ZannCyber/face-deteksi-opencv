import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def calculate_ear(eye_landmarks):
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    vert1 = abs(p2.y - p6.y)
    vert2 = abs(p3.y - p5.y)
    horiz = abs(p1.x - p4.x)
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

def count_fingers(hand_landmarks, hand_label):
    fingers = 0
    if hand_label == 'Left':
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x:
            fingers += 1
    else:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
            fingers += 1
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers += 1
    return fingers

def is_thumbs_up(hand_landmarks, hand_label):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_mcp = hand_landmarks.landmark[5]
    if hand_label == 'Left':
        if thumb_tip.x > thumb_ip.x and all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18])):
            return True
    else:
        if thumb_tip.x < thumb_ip.x and all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18])):
            return True
    return False

def is_peace_sign(hand_landmarks):
    if (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
        all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y for tip, pip in zip([16, 20], [14, 18]))):
        return True
    return False

def is_clasped_hands(hand_landmarks_list):
    if len(hand_landmarks_list) == 2:
        left_palm = None
        right_palm = None
        for hand_landmarks, hand_label in hand_landmarks_list:
            if hand_label == 'Left':
                left_palm = hand_landmarks.landmark[0]
            elif hand_label == 'Right':
                right_palm = hand_landmarks.landmark[0]
        if left_palm and right_palm:
            dist = math.hypot(left_palm.x - right_palm.x, left_palm.y - right_palm.y)
            if dist < 0.05:
                return True
    return False

def detect_smile(face_landmarks):
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    upper_lip = face_landmarks.landmark[0]
    lower_lip = face_landmarks.landmark[17]
    mouth_width = abs(left_mouth.x - right_mouth.x)
    mouth_height = abs(upper_lip.y - lower_lip.y)
    mar = mouth_height / mouth_width
    if mar < 0.2 and abs(left_mouth.y - upper_lip.y) < 0.02 and abs(right_mouth.y - upper_lip.y) < 0.02:
        return False
    left_corner_up = face_landmarks.landmark[61].y < face_landmarks.landmark[0].y
    right_corner_up = face_landmarks.landmark[291].y < face_landmarks.landmark[0].y
    if left_corner_up and right_corner_up:
        return True
    return False

def detect_angry(face_landmarks):
    left_brow = face_landmarks.landmark[70]
    right_brow = face_landmarks.landmark[300]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    brow_left_dist = abs(left_brow.y - left_eye.y)
    brow_right_dist = abs(right_brow.y - right_eye.y)
    if brow_left_dist < 0.05 and brow_right_dist < 0.05:
        return True
    return False

def estimate_age(face_landmarks, frame_shape):
    landmarks = face_landmarks.landmark
    face_width = abs(landmarks[234].x - landmarks[454].x) * frame_shape[1]
    face_height = abs(landmarks[10].y - landmarks[152].y) * frame_shape[0]
    face_area = face_width * face_height
    if face_area > 0.3 * frame_shape[0] * frame_shape[1]:
        return "Muda (di bawah 30)"
    elif face_area > 0.15 * frame_shape[0] * frame_shape[1]:
        return "Dewasa (30-50)"
    else:
        return "Tua (di atas 50)"

def get_face_bounding_box(face_landmarks, frame_shape):
    landmarks = face_landmarks.landmark
    x_coords = [landmark.x * frame_shape[1] for landmark in landmarks]
    y_coords = [landmark.y * frame_shape[0] for landmark in landmarks]
    x_min = int(min(x_coords)) - 20
    y_min = int(min(y_coords)) - 20
    x_max = int(max(x_coords)) + 20
    y_max = int(max(y_coords)) + 20
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_shape[1], x_max)
    y_max = min(frame_shape[0], y_max)
    return (x_min, y_min, x_max, y_max)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
text_color = (255, 255, 255)
text_start_y = 30
line_spacing = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_shape = frame.shape

    texts = []

    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            left_eye_lm = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye_lm = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            ear_left = calculate_ear(left_eye_lm)
            ear_right = calculate_ear(right_eye_lm)
            ear_avg = (ear_left + ear_right) / 2.0
            if ear_avg < 0.25:
                texts.append(("Ngantuk!", (0, 0, 255)))
            if detect_smile(face_landmarks):
                texts.append(("Senyum!", (0, 255, 0)))
            if detect_angry(face_landmarks):
                texts.append(("Marah!", (0, 0, 255)))
            age = estimate_age(face_landmarks, frame_shape)
            texts.append((f"Usia: {age}", (0, 255, 255)))
            x_min, y_min, x_max, y_max = get_face_bounding_box(face_landmarks, frame_shape)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_draw.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
    else:
        texts.append(("Muka tidak terdeteksi", (0, 0, 255)))

    hand_results = hands.process(frame_rgb)
    total_fingers = 0
    thumbs_up = False
    peace = False
    clasped = False
    hand_landmarks_list = []
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_label = handedness.classification[0].label
            hand_landmarks_list.append((hand_landmarks, hand_label))
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            total_fingers += count_fingers(hand_landmarks, hand_label)
            if is_thumbs_up(hand_landmarks, hand_label):
                thumbs_up = True
            if is_peace_sign(hand_landmarks):
                peace = True
    
    if hand_landmarks_list:
        if is_clasped_hands(hand_landmarks_list):
            texts.append(("Minta Maaf!", (255, 165, 0)))
        if thumbs_up:
            texts.append(("Bagus!", (0, 255, 0)))
        if peace:
            texts.append(("Peace!", (255, 255, 0)))
        if total_fingers == 5:
            texts.append(("Hai", (255, 0, 0)))
        elif total_fingers == 10:
            texts.append(("Halo semua", (255, 0, 0)))
        elif total_fingers > 0:
            texts.append((f"{total_fingers} jari", (255, 0, 0)))

    current_y = text_start_y
    for text, color in texts:
        cv2.putText(frame, text, (20, current_y), font, font_scale, color, font_thickness)
        current_y += line_spacing

    cv2.imshow('Deteksi Ngantuk, Gesture, dan Emosi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()