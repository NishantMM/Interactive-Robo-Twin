import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Physics Variables ---
dot_pos = [300, 300]
velocity_y = 0
gravity = 1.5  # The strength of the pull
friction = 0.7  # How much it bounces (0.7 = 70% energy kept)
is_grabbing = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    robot_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Get Thumb(4) and Index(8)
            tx, ty = int(hand_lms.landmark[4].x * w), int(hand_lms.landmark[4].y * h)
            ix, iy = int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h)
            pinch_x, pinch_y = (tx + ix) // 2, (ty + iy) // 2

            finger_dist = math.hypot(tx - ix, ty - iy)
            target_dist = math.hypot(pinch_x - dot_pos[0], pinch_y - dot_pos[1])

            # Grabbing Logic
            if finger_dist < 40 and target_dist < 50:
                is_grabbing = True
                velocity_y = 0  # Stop falling while held
                dot_pos = [pinch_x, pinch_y]
            else:
                is_grabbing = False

            # Draw Skeleton
            mp_draw.draw_landmarks(robot_canvas, hand_lms, mp_hands.HAND_CONNECTIONS)

    # --- Physics Engine Update ---
    if not is_grabbing:
        velocity_y += gravity  # Apply gravity
        dot_pos[1] += int(velocity_y)  # Move dot down

        # Floor Collision (Don't let it fall past the bottom)
        if dot_pos[1] > h - 20:
            dot_pos[1] = h - 20
            velocity_y = -velocity_y * friction  # Bounce!

            # Stop tiny jitters when it's almost still
            if abs(velocity_y) < 2: velocity_y = 0

    # Draw the Object
    color = (0, 255, 0) if is_grabbing else (0, 0, 255)
    cv2.circle(robot_canvas, (dot_pos[0], dot_pos[1]), 20, color, cv2.FILLED)

    # Visual Output
    combined = np.hstack((frame, robot_canvas))
    cv2.imshow("Robot Physics Mimicry", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
