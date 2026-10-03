# ============================================================
# SKIN FINDING
# Real-Time Hand Gesture Recognition Using Skin Color Models
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk

from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from pathlib import Path
from datetime import datetime
from collections import deque, Counter

import time


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hand_landmarker.task"

# Results folder
RESULTS_DIR = BASE_DIR / "results"

# If results exists as a FILE, use another folder
if RESULTS_DIR.exists() and not RESULTS_DIR.is_dir():
    RESULTS_DIR = BASE_DIR / "results_images"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "\nhand_landmarker.task was not found.\n\n"
        "Place hand_landmarker.task inside:\n"
        f"{BASE_DIR}\n"
    )


# ============================================================
# 3. MEDIAPIPE TASKS API
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


# ============================================================
# 4. MEDIAPIPE CONFIGURATION
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.65
)


# ============================================================
# 5. CREATE HAND LANDMARKER
# ============================================================

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# 6. COLORS
# ============================================================

BG = "#0B1117"
HEADER = "#111A22"
PANEL = "#151F28"
CARD = "#202C37"

WHITE = "#F4F7FA"
GRAY = "#91A0AD"

GREEN = "#35D07F"
RED = "#FF5C5C"
BLUE = "#4DA3FF"
CYAN = "#48D7E8"
YELLOW = "#FFD166"


# ============================================================
# 7. GLOBAL VARIABLES
# ============================================================

cap = None

camera_running = False

latest_frame = None

latest_processed_frame = None

finger_history = deque(maxlen=7)

fps = 0

previous_time = time.time()


# ============================================================
# 8. HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# 9. DISTANCE FUNCTION
# ============================================================

def landmark_distance(a, b):
    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


# ============================================================
# 10. FINGER COUNT
# ============================================================

def count_fingers(landmarks):

    wrist = landmarks[0]

    count = 0

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    thumb_tip_dist = landmark_distance(
        thumb_tip,
        wrist
    )

    thumb_ip_dist = landmark_distance(
        thumb_ip,
        wrist
    )

    thumb_mcp_dist = landmark_distance(
        thumb_mcp,
        wrist
    )

    if (
        thumb_tip_dist > thumb_ip_dist * 1.08
        and
        thumb_tip_dist > thumb_mcp_dist * 1.20
    ):
        count += 1


    # --------------------------------------------------------
    # OTHER FOUR FINGERS
    # --------------------------------------------------------

    fingers = [
        (8, 6, 5),      # Index
        (12, 10, 9),    # Middle
        (16, 14, 13),   # Ring
        (20, 18, 17)    # Pinky
    ]

    for tip_id, pip_id, mcp_id in fingers:

        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        mcp = landmarks[mcp_id]

        tip_dist = landmark_distance(
            tip,
            wrist
        )

        pip_dist = landmark_distance(
            pip,
            wrist
        )

        mcp_dist = landmark_distance(
            mcp,
            wrist
        )

        if (
            tip_dist > pip_dist * 1.08
            and
            tip_dist > mcp_dist * 1.25
        ):
            count += 1

    return count


# ============================================================
# 11. STABLE FINGER COUNT
# ============================================================

def get_stable_count(value):

    finger_history.append(value)

    if len(finger_history) < 3:
        return value

    counter = Counter(finger_history)

    return counter.most_common(1)[0][0]


# ============================================================
# 12. GESTURE NAME
# ============================================================

def get_gesture(count):

    gestures = {
        0: "FIST",
        1: "ONE FINGER",
        2: "TWO FINGERS",
        3: "THREE FINGERS",
        4: "FOUR FINGERS",
        5: "OPEN HAND"
    }

    return gestures.get(
        count,
        "UNKNOWN"
    )


# ============================================================
# 13. ACTION NAME
# ============================================================

def get_action(count):

    actions = {
        0: "STOP",
        1: "SELECT",
        2: "NEXT",
        3: "PREVIOUS",
        4: "NO ACTION",
        5: "START"
    }

    return actions.get(
        count,
        "NO ACTION"
    )


# ============================================================
# 14. HSV SKIN DETECTION
# ============================================================

def create_skin_mask(frame):

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

    mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# 15. DRAW HAND
# ============================================================

def draw_hand(frame, landmarks):

    height, width, _ = frame.shape

    points = []

    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        x = max(
            0,
            min(width - 1, x)
        )

        y = max(
            0,
            min(height - 1, y)
        )

        points.append(
            (x, y)
        )

        cv2.circle(
            frame,
            (x, y),
            5,
            (53, 208, 127),
            -1
        )

    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (72, 215, 232),
            2
        )

    return points


# ============================================================
# 16. PROCESS CAMERA FRAME
# ============================================================

def process_frame(frame):

    global fps
    global previous_time

    display = frame.copy()

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:
        current_fps = 1.0 / elapsed

        fps = int(
            current_fps
        )

    previous_time = current_time

    # --------------------------------------------------------
    # HSV MASK
    # --------------------------------------------------------

    skin_mask = create_skin_mask(
        frame
    )

    total_pixels = (
        skin_mask.shape[0] *
        skin_mask.shape[1]
    )

    skin_pixels = cv2.countNonZero(
        skin_mask
    )

    skin_percentage = (
        skin_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    finger_count = 0

    gesture = "NO HAND"

    action = "NO ACTION"

    handedness = "N/A"

    confidence = 0.0

    # --------------------------------------------------------
    # Hand detected
    # --------------------------------------------------------

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # Finger count
        raw_count = count_fingers(
            landmarks
        )

        finger_count = get_stable_count(
            raw_count
        )

        # Gesture
        gesture = get_gesture(
            finger_count
        )

        # Action
        action = get_action(
            finger_count
        )

        # ----------------------------------------------------
        # Handedness
        # ----------------------------------------------------

        if result.handedness:

            hand_info = (
                result.handedness[0][0]
            )

            handedness = (
                hand_info.category_name
            )

            confidence = (
                hand_info.score
            )

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        points = draw_hand(
            display,
            landmarks
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        xs = [
            p[0]
            for p in points
        ]

        ys = [
            p[1]
            for p in points
        ]

        x1 = max(
            min(xs) - 20,
            0
        )

        y1 = max(
            min(ys) - 20,
            0
        )

        x2 = min(
            max(xs) + 20,
            display.shape[1] - 1
        )

        y2 = min(
            max(ys) + 20,
            display.shape[0] - 1
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (53, 208, 127),
            2
        )

        # ----------------------------------------------------
        # Text on camera
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Fingers: {finger_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (53, 208, 127),
            2
        )

        cv2.putText(
            display,
            f"Gesture: {gesture}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (72, 215, 232),
            2
        )

        cv2.putText(
            display,
            f"Action: {action}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 209, 102),
            2
        )

    else:

        # ----------------------------------------------------
        # No hand
        # ----------------------------------------------------

        finger_history.clear()

        cv2.putText(
            display,
            "NO HAND DETECTED",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 92, 92),
            2
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"FPS: {fps}",
        (15, display.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Mask display
    # --------------------------------------------------------

    mask_display = cv2.cvtColor(
        skin_mask,
        cv2.COLOR_GRAY2BGR
    )

    return (
        display,
        mask_display,
        finger_count,
        gesture,
        action,
        handedness,
        confidence,
        skin_percentage
    )


# ============================================================
# 17. START CAMERA
# ============================================================

def start_camera():

    global cap
    global camera_running
    global previous_time

    if camera_running:
        return

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        cap.release()

        cap = None

        messagebox.showerror(
            "Camera Error",
            "Camera could not be opened.\n\n"
            "Check whether another application is using "
            "the webcam."
        )

        return

    # Camera resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    camera_running = True

    previous_time = time.time()

    finger_history.clear()

    status_label.config(
        text="● LIVE",
        fg=GREEN
    )

    start_button.config(
        state="disabled"
    )

    stop_button.config(
        state="normal"
    )

    update_camera()


# ============================================================
# 18. STOP CAMERA
# ============================================================

def stop_camera():

    global cap
    global camera_running

    camera_running = False

    if cap is not None:

        cap.release()

        cap = None

    status_label.config(
        text="● STOPPED",
        fg=RED
    )

    start_button.config(
        state="normal"
    )

    stop_button.config(
        state="disabled"
    )

    finger_history.clear()

    reset_dashboard()


# ============================================================
# 19. UPDATE CAMERA
# ============================================================

def update_camera():

    global latest_frame
    global latest_processed_frame

    if not camera_running:
        return

    if cap is None:
        return

    ret, frame = cap.read()

    if not ret:

        root.after(
            30,
            update_camera
        )

        return

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    latest_frame = frame.copy()

    (
        processed,
        mask,
        finger_count,
        gesture,
        action,
        handedness,
        confidence,
        skin_percentage
    ) = process_frame(frame)

    latest_processed_frame = (
        processed.copy()
    )

    # ========================================================
    # CAMERA IMAGE
    # ========================================================

    camera_rgb = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB
    )

    camera_image = Image.fromarray(
        camera_rgb
    )

    # Fit camera image into panel
    camera_image.thumbnail(
        (560, 420),
        Image.Resampling.LANCZOS
    )

    camera_photo = ImageTk.PhotoImage(
        camera_image
    )

    camera_label.config(
        image=camera_photo
    )

    camera_label.image = camera_photo

    # ========================================================
    # MASK IMAGE
    # ========================================================

    mask_rgb = cv2.cvtColor(
        mask,
        cv2.COLOR_BGR2RGB
    )

    mask_image = Image.fromarray(
        mask_rgb
    )

    mask_image.thumbnail(
        (260, 420),
        Image.Resampling.LANCZOS
    )

    mask_photo = ImageTk.PhotoImage(
        mask_image
    )

    mask_label.config(
        image=mask_photo
    )

    mask_label.image = mask_photo

    # ========================================================
    # DASHBOARD UPDATE
    # ========================================================

    finger_value.config(
        text=str(finger_count)
    )

    gesture_value.config(
        text=gesture
    )

    action_value.config(
        text=action
    )

    hand_value.config(
        text=handedness
    )

    confidence_value.config(
        text=f"{confidence * 100:.1f}%"
    )

    fps_value.config(
        text=str(fps)
    )

    skin_value.config(
        text=f"{skin_percentage:.1f}%"
    )

    # ========================================================
    # NEXT FRAME
    # ========================================================

    root.after(
        25,
        update_camera
    )


# ============================================================
# 20. CAPTURE IMAGE
# ============================================================

def capture_image():

    if latest_processed_frame is None:

        messagebox.showwarning(
            "Capture",
            "Please start the camera first."
        )

        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        RESULTS_DIR /
        f"gesture_{timestamp}.jpg"
    )

    success = cv2.imwrite(
        str(filename),
        latest_processed_frame
    )

    if success:

        messagebox.showinfo(
            "Capture Successful",
            "Gesture image saved successfully.\n\n"
            f"{filename}"
        )

    else:

        messagebox.showerror(
            "Capture Error",
            "Unable to save the image."
        )


# ============================================================
# 21. RESET DASHBOARD
# ============================================================

def reset_dashboard():

    finger_history.clear()

    finger_value.config(
        text="0"
    )

    gesture_value.config(
        text="NO HAND"
    )

    action_value.config(
        text="NO ACTION"
    )

    hand_value.config(
        text="N/A"
    )

    confidence_value.config(
        text="0%"
    )

    fps_value.config(
        text="0"
    )

    skin_value.config(
        text="0%" 
    )


# ============================================================
# 22. EXIT APPLICATION
# ============================================================

def exit_application():

    global cap
    global camera_running

    camera_running = False

    if cap is not None:

        cap.release()

        cap = None

    try:
        landmarker.close()
    except Exception:
        pass

    root.destroy()


# ============================================================
# 23. CREATE MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Skin Finding - Real-Time Hand Gesture Recognition"
)

# IMPORTANT:
# This size is intentionally smaller so the complete GUI
# including buttons fits on normal Windows screens.

root.geometry(
    "1150x650"
)

root.minsize(
    1000,
    600
)

root.configure(
    bg=BG
)


# ============================================================
# 24. ROOT GRID
# ============================================================

root.grid_rowconfigure(
    0,
    weight=0
)

root.grid_rowconfigure(
    1,
    weight=1
)

root.grid_rowconfigure(
    2,
    weight=0
)

root.grid_columnconfigure(
    0,
    weight=1
)


# ============================================================
# 25. HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=HEADER,
    height=70
)

header.grid(
    row=0,
    column=0,
    sticky="ew"
)

header.grid_propagate(
    False
)


title_label = tk.Label(
    header,
    text="SKIN FINDING",
    font=("Segoe UI", 21, "bold"),
    bg=HEADER,
    fg=WHITE
)

title_label.pack(
    pady=(5, 0)
)


subtitle_label = tk.Label(
    header,
    text="Real-Time Hand Gesture Recognition Using Skin Color Models",
    font=("Segoe UI", 9),
    bg=HEADER,
    fg=GRAY
)

subtitle_label.pack()


status_label = tk.Label(
    header,
    text="● STOPPED",
    font=("Segoe UI", 9, "bold"),
    bg=HEADER,
    fg=RED
)

status_label.place(
    relx=0.94,
    rely=0.50,
    anchor="center"
)


# ============================================================
# 26. MAIN CONTENT
# ============================================================

content = tk.Frame(
    root,
    bg=BG
)

content.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=8,
    pady=7
)

content.grid_rowconfigure(
    0,
    weight=1
)

content.grid_columnconfigure(
    0,
    weight=5
)

content.grid_columnconfigure(
    1,
    weight=2
)

content.grid_columnconfigure(
    2,
    weight=2
)


# ============================================================
# 27. CAMERA PANEL
# ============================================================

camera_panel = tk.Frame(
    content,
    bg=PANEL,
    bd=1,
    relief="solid"
)

camera_panel.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 4)
)

camera_panel.grid_rowconfigure(
    1,
    weight=1
)

camera_panel.grid_columnconfigure(
    0,
    weight=1
)


camera_heading = tk.Label(
    camera_panel,
    text="LIVE CAMERA",
    font=("Segoe UI", 12, "bold"),
    bg=PANEL,
    fg=WHITE
)

camera_heading.grid(
    row=0,
    column=0,
    pady=6
)


camera_label = tk.Label(
    camera_panel,
    bg="black",
    text="Camera stopped",
    font=("Segoe UI", 10),
    fg=GRAY
)

camera_label.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=7,
    pady=3
)


camera_info = tk.Label(
    camera_panel,
    text="Landmarks • Bounding Box • Gesture",
    font=("Segoe UI", 8),
    bg=PANEL,
    fg=GRAY
)

camera_info.grid(
    row=2,
    column=0,
    pady=5
)


# ============================================================
# 28. HSV MASK PANEL
# ============================================================

mask_panel = tk.Frame(
    content,
    bg=PANEL,
    bd=1,
    relief="solid"
)

mask_panel.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=4
)

mask_panel.grid_rowconfigure(
    1,
    weight=1
)

mask_panel.grid_columnconfigure(
    0,
    weight=1
)


mask_heading = tk.Label(
    mask_panel,
    text="HSV SKIN MASK",
    font=("Segoe UI", 12, "bold"),
    bg=PANEL,
    fg=WHITE
)

mask_heading.grid(
    row=0,
    column=0,
    pady=6
)


mask_label = tk.Label(
    mask_panel,
    bg="black",
    text="Mask",
    font=("Segoe UI", 10),
    fg=GRAY
)

mask_label.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=6,
    pady=3
)


mask_info = tk.Label(
    mask_panel,
    text="Detected skin region",
    font=("Segoe UI", 8),
    bg=PANEL,
    fg=GRAY
)

mask_info.grid(
    row=2,
    column=0,
    pady=5
)


# ============================================================
# 29. DASHBOARD PANEL
# ============================================================

dashboard = tk.Frame(
    content,
    bg=PANEL,
    bd=1,
    relief="solid"
)

dashboard.grid(
    row=0,
    column=2,
    sticky="nsew",
    padx=(4, 0)
)


dashboard_heading = tk.Label(
    dashboard,
    text="GESTURE DASHBOARD",
    font=("Segoe UI", 11, "bold"),
    bg=PANEL,
    fg=WHITE
)

dashboard_heading.pack(
    pady=(7, 4)
)


# ============================================================
# 30. DASHBOARD CARD FUNCTION
# ============================================================

def create_dashboard_card(
    parent,
    title,
    value,
    value_color=WHITE
):

    card = tk.Frame(
        parent,
        bg=CARD,
        height=36
    )

    card.pack(
        fill="x",
        padx=7,
        pady=2
    )

    card.pack_propagate(
        False
    )


    title_widget = tk.Label(
        card,
        text=title,
        font=("Segoe UI", 7, "bold"),
        bg=CARD,
        fg=GRAY,
        anchor="w"
    )

    title_widget.pack(
        side="left",
        padx=6
    )


    value_widget = tk.Label(
        card,
        text=value,
        font=("Segoe UI", 8, "bold"),
        bg=CARD,
        fg=value_color,
        anchor="e"
    )

    value_widget.pack(
        side="right",
        padx=6
    )


    return value_widget


# ============================================================
# 31. DASHBOARD VALUES
# ============================================================

finger_value = create_dashboard_card(
    dashboard,
    "FINGER COUNT",
    "0",
    GREEN
)


gesture_value = create_dashboard_card(
    dashboard,
    "GESTURE",
    "NO HAND",
    CYAN
)


action_value = create_dashboard_card(
    dashboard,
    "ACTION",
    "NO ACTION",
    YELLOW
)


hand_value = create_dashboard_card(
    dashboard,
    "HANDEDNESS",
    "N/A",
    WHITE
)


confidence_value = create_dashboard_card(
    dashboard,
    "CONFIDENCE",
    "0%",
    GREEN
)


fps_value = create_dashboard_card(
    dashboard,
    "FPS",
    "0",
    BLUE
)


skin_value = create_dashboard_card(
    dashboard,
    "SKIN COVERAGE",
    "0%",
    YELLOW
)


# ============================================================
# 32. MAPPING
# ============================================================

mapping_heading = tk.Label(
    dashboard,
    text="GESTURE MAPPING",
    font=("Segoe UI", 9, "bold"),
    bg=PANEL,
    fg=YELLOW
)

mapping_heading.pack(
    pady=(5, 2)
)


mapping_frame = tk.Frame(
    dashboard,
    bg=CARD
)

mapping_frame.pack(
    fill="x",
    padx=7,
    pady=2
)


mapping_data = [
    ("0", "STOP"),
    ("1", "SELECT"),
    ("2", "NEXT"),
    ("3", "PREVIOUS"),
    ("4", "NO ACTION"),
    ("5", "START")
]


for number, action_name in mapping_data:

    row = tk.Frame(
        mapping_frame,
        bg=CARD
    )

    row.pack(
        fill="x",
        padx=5,
        pady=1
    )


    tk.Label(
        row,
        text=number,
        font=("Segoe UI", 7, "bold"),
        bg=CARD,
        fg=GREEN,
        width=2,
        anchor="w"
    ).pack(
        side="left"
    )


    tk.Label(
        row,
        text="→",
        font=("Segoe UI", 7),
        bg=CARD,
        fg=GRAY
    ).pack(
        side="left"
    )


    tk.Label(
        row,
        text=action_name,
        font=("Segoe UI", 7),
        bg=CARD,
        fg=WHITE,
        anchor="w"
    ).pack(
        side="left",
        padx=3
    )


# ============================================================
# 33. CONTROL BAR
# ============================================================

# This row is completely separate from the camera area.
# Therefore the buttons cannot be pushed below the window.

control_bar = tk.Frame(
    root,
    bg=HEADER,
    height=62
)

control_bar.grid(
    row=2,
    column=0,
    sticky="ew"
)

control_bar.grid_propagate(
    False
)


# ============================================================
# 34. BUTTON CONTAINER
# ============================================================

button_container = tk.Frame(
    control_bar,
    bg=HEADER
)

button_container.pack(
    pady=10
)


# ============================================================
# 35. BUTTON STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except Exception:
    pass


style.configure(
    "Control.TButton",
    font=("Segoe UI", 9, "bold"),
    padding=(14, 7)
)


# ============================================================
# 36. START BUTTON
# ============================================================

start_button = ttk.Button(
    button_container,
    text="▶ START CAMERA",
    style="Control.TButton",
    command=start_camera
)

start_button.pack(
    side="left",
    padx=4
)


# ============================================================
# 37. STOP BUTTON
# ============================================================

stop_button = ttk.Button(
    button_container,
    text="■ STOP CAMERA",
    style="Control.TButton",
    command=stop_camera,
    state="disabled"
)

stop_button.pack(
    side="left",
    padx=4
)


# ============================================================
# 38. CAPTURE BUTTON
# ============================================================

capture_button = ttk.Button(
    button_container,
    text="● CAPTURE",
    style="Control.TButton",
    command=capture_image
)

capture_button.pack(
    side="left",
    padx=4
)


# ============================================================
# 39. RESET BUTTON
# ============================================================

reset_button = ttk.Button(
    button_container,
    text="↻ RESET",
    style="Control.TButton",
    command=reset_dashboard
)

reset_button.pack(
    side="left",
    padx=4
)


# ============================================================
# 40. EXIT BUTTON
# ============================================================

exit_button = ttk.Button(
    button_container,
    text="✕ EXIT",
    style="Control.TButton",
    command=exit_application
)

exit_button.pack(
    side="left",
    padx=4
)


# ============================================================
# 41. WINDOW CLOSE
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    exit_application
)


# ============================================================
# 42. START PROGRAM
# ============================================================

root.mainloop()