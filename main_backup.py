import cv2
import mediapipe as mp
import numpy as np
import math

# =========================================================
# 1. INITIALIZE WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# =========================================================
# 2. INITIALIZE MEDIAPIPE HAND LANDMARKER
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)


# =========================================================
# 3. DISTANCE FUNCTION
# =========================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# =========================================================
# 4. FINGER COUNTING
# =========================================================

def count_fingers(hand):

    count = 0

    # Index finger
    if hand[8].y < hand[6].y:
        count += 1

    # Middle finger
    if hand[12].y < hand[10].y:
        count += 1

    # Ring finger
    if hand[16].y < hand[14].y:
        count += 1

    # Pinky
    if hand[20].y < hand[18].y:
        count += 1

    # Thumb
    wrist = hand[0]
    thumb_tip = hand[4]
    thumb_joint = hand[3]

    thumb_tip_distance = distance(
        wrist,
        thumb_tip
    )

    thumb_joint_distance = distance(
        wrist,
        thumb_joint
    )

    if thumb_tip_distance > thumb_joint_distance * 1.15:
        count += 1

    return count


# =========================================================
# 5. GESTURE RECOGNITION
# =========================================================

def recognize_gesture(count):

    if count == 0:
        return "Fist"

    elif count == 1:
        return "One Finger"

    elif count == 2:
        return "Two Fingers"

    elif count == 3:
        return "Three Fingers"

    elif count == 4:
        return "Four Fingers"

    elif count == 5:
        return "Open Hand"

    return "Unknown"


# =========================================================
# 6. INTERACTIVE ACTION
# =========================================================

def gesture_action(count):

    if count == 0:
        return "STOP"

    elif count == 1:
        return "SELECT"

    elif count == 2:
        return "NEXT"

    elif count == 3:
        return "PREVIOUS"

    elif count == 5:
        return "START"

    else:
        return "NO ACTION"


# =========================================================
# 7. DRAW HAND LANDMARKS
# =========================================================

def draw_hand(frame, hand):

    height, width = frame.shape[:2]

    # Draw landmark points
    for landmark in hand:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Hand connections
    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (0, 17)
    ]

    for start, end in connections:

        x1 = int(
            hand[start].x * width
        )

        y1 = int(
            hand[start].y * height
        )

        x2 = int(
            hand[end].x * width
        )

        y2 = int(
            hand[end].y * height
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


# =========================================================
# 8. MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]


    # =====================================================
    # 9. HSV SKIN COLOR DETECTION
    # =====================================================

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    lower_skin = np.array(
        [0, 20, 70],
        dtype=np.uint8
    )

    upper_skin = np.array(
        [20, 255, 255],
        dtype=np.uint8
    )

    skin_mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )


    # =====================================================
    # 10. REMOVE NOISE
    # =====================================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # =====================================================
    # 11. CONVERT FRAME FOR MEDIAPIPE
    # =====================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # =====================================================
    # 12. DETECT HAND
    # =====================================================

    result = landmarker.detect(
        mp_image
    )

    finger_count = 0
    gesture = "No Hand"
    action = "NO ACTION"


    # =====================================================
    # 13. PROCESS HAND
    # =====================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Count fingers
        finger_count = count_fingers(
            hand
        )

        # Recognize gesture
        gesture = recognize_gesture(
            finger_count
        )

        # Get action
        action = gesture_action(
            finger_count
        )

        # Draw landmarks
        draw_hand(
            frame,
            hand
        )


        # =================================================
        # 14. DRAW HAND BOUNDING BOX
        # =================================================

        x_values = [
            int(point.x * width)
            for point in hand
        ]

        y_values = [
            int(point.y * height)
            for point in hand
        ]

        x_min = max(
            min(x_values) - 20,
            0
        )

        x_max = min(
            max(x_values) + 20,
            width - 1
        )

        y_min = max(
            min(y_values) - 20,
            0
        )

        y_max = min(
            max(y_values) + 20,
            height - 1
        )

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (255, 255, 0),
            2
        )


    # =====================================================
    # 15. DISPLAY PROJECT INFORMATION
    # =====================================================

    cv2.putText(
        frame,
        "HSV Skin Detection: ON",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Fingers: " + str(finger_count),
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Action: " + action,
        (25, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Skin Finding - Hand Gesture Recognition",
        (25, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # =====================================================
    # 16. SKIN MASK PREVIEW
    # =====================================================

    skin_preview = cv2.cvtColor(
        skin_mask,
        cv2.COLOR_GRAY2BGR
    )

    skin_preview = cv2.resize(
        skin_preview,
        (240, 180)
    )

    cv2.rectangle(
        skin_preview,
        (0, 0),
        (239, 179),
        (255, 255, 255),
        2
    )

    if height >= 190 and width >= 250:

        frame[
            10:190,
            width - 250:width - 10
        ] = skin_preview


    # =====================================================
    # 17. SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "Real-Time Hand Gesture Recognition",
        frame
    )


    # =====================================================
    # 18. PRESS Q TO EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# 19. RELEASE RESOURCES
# =========================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()