
import pyautogui
import time
import math

# === PARAMETRY ===
ACTIVATION_SECONDS = 4.0
CURSOR_SMOOTHING = 0.5

# === STAN ===
mouse_control_active = False
activation_start_time = None
last_cursor_pos = None
clicking = False

def distance(a, b):
    return math.sqrt(sum((ax - bx) ** 2 for ax, bx in zip(a, b)))

def is_fist(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    folded = 0
    wrist = hand_landmarks[0]
    for idx in tips_ids:
        if distance(hand_landmarks[idx], wrist) < 0.1:
            folded += 1
    return folded >= 4

def is_hand_flat(hand_landmarks):
    tips = [hand_landmarks[i] for i in [8, 12, 16, 20]]
    distances = [distance(tips[i], tips[i+1]) for i in range(len(tips)-1)]
    total_spread = sum(distances)
    avg_dist_to_wrist = sum(distance(tip, hand_landmarks[0]) for tip in tips) / len(tips)
    return total_spread < 0.2 and avg_dist_to_wrist > 0.2

def is_near_face(hand_landmarks):
    cx, cy = hand_landmarks[0][0], hand_landmarks[0][1]
    return 0.35 < cx < 0.65 and 0.2 < cy < 0.6

def mouse_control_from_gestures(left_hand, right_hand):
    global mouse_control_active, activation_start_time, last_cursor_pos, clicking

    # === AKTYWACJA MYSZKI LEWĄ RĘKĄ ===
    if left_hand and is_fist(left_hand) and is_near_face(left_hand):
        if activation_start_time is None:
            activation_start_time = time.time()
        elif time.time() - activation_start_time >= ACTIVATION_SECONDS:
            if not mouse_control_active:
                print("🖱️ Tryb myszki: AKTYWNY")
                mouse_control_active = True
    else:
        activation_start_time = None
        if mouse_control_active:
            print("⛔ Tryb myszki: WYŁĄCZONY")
        mouse_control_active = False
        clicking = False
        return "Tryb myszki: WYŁĄCZONY"

    # === STEROWANIE PRAWĄ RĘKĄ ===
    if mouse_control_active and right_hand:
        if is_fist(right_hand):
            if not clicking:
                pyautogui.mouseDown()
                clicking = True
        elif is_hand_flat(right_hand):
            x = right_hand[9][0]
            y = right_hand[9][1]
            screen_w, screen_h = pyautogui.size()
            new_x = int(screen_w * x)
            new_y = int(screen_h * y)

            if last_cursor_pos:
                new_x = int(last_cursor_pos[0] * (1 - CURSOR_SMOOTHING) + new_x * CURSOR_SMOOTHING)
                new_y = int(last_cursor_pos[1] * (1 - CURSOR_SMOOTHING) + new_y * CURSOR_SMOOTHING)

            pyautogui.moveTo(new_x, new_y)
            last_cursor_pos = (new_x, new_y)
            clicking = False
        else:
            if clicking:
                pyautogui.mouseUp()
                clicking = False

    return "Tryb myszki: AKTYWNY" if mouse_control_active else "Tryb myszki: WYŁĄCZONY"
