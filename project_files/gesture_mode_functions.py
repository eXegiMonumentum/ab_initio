import cv2
import numpy as np
from collections import deque
import time

# Parametry
FPS = 20
COUNTDOWN_FRAMES = FPS * 3
STABLE_FRAMES = FPS // 2
gesture_state = {
    "current": None,
    "stable_counter": 0,
    "countdown_active": False,
    "countdown": 0,
    "timer": 0,
    "last_detected": "",
    "last_frame": -100,
    "display_frames": FPS * 2
}

gesture_counter = {
    "ZOOM_IN": 0,
    "ZOOM_OUT": 0,
    "STOP": 0
}

# Bufor do porównania pozycji
recent_frames = deque(maxlen=STABLE_FRAMES)

# === Heurystyki ===
def is_fist(landmarks):
    folded = lambda tip, pip: landmarks[tip].y > landmarks[pip].y
    return all(folded(t, p) for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)])

def is_flat_hand(landmarks):
    extended = lambda tip, pip: landmarks[tip].y < landmarks[pip].y
    return all(extended(t, p) for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)])

def hand_center(landmarks):
    return np.mean([[lm.x, lm.y] for lm in landmarks], axis=0)

# === Gesty ===
def detect_zoom_in(frame):
    if len(frame) != 2: return False
    left, right = None, None
    for hand in frame:
        if hand["type"] == "Left": left = hand
        if hand["type"] == "Right": right = hand
    if not left or not right: return False
    if not is_fist(left["landmarks"]) or not is_fist(right["landmarks"]): return False
    lp = hand_center(left["landmarks"])
    rp = hand_center(right["landmarks"])
    dist = np.linalg.norm(lp - rp)
    if dist > 0.15 or abs(lp[1] - rp[1]) > 0.05: return False
    return True

def detect_zoom_out(frame):
    if len(frame) != 2: return False
    left, right = None, None
    for hand in frame:
        if hand["type"] == "Left": left = hand
        if hand["type"] == "Right": right = hand
    if not left or not right: return False
    if not is_fist(left["landmarks"]) or not is_fist(right["landmarks"]): return False
    lp = hand_center(left["landmarks"])
    rp = hand_center(right["landmarks"])
    spread_x = abs(lp[0] - rp[0])
    delta_y = abs(lp[1] - rp[1])
    if spread_x < 0.35 or delta_y > 0.05: return False
    return True

def detect_stop(frame):
    if len(frame) != 1: return False
    hand = frame[0]
    if hand["type"] != "Left": return False
    if not is_flat_hand(hand["landmarks"]): return False
    center = hand_center(hand["landmarks"])
    if center[0] > 0.2: return False  # dłoń musi być wystawiona dalej w lewo
    return True

# === Główna funkcja ===
def run_gesture_mode(frame, hands_data):
    global recent_frames, gesture_state, gesture_counter

    recent_frames.append(hands_data)

    gesture_now = None
    if detect_zoom_in(hands_data):
        gesture_now = "ZOOM_IN"
    elif detect_zoom_out(hands_data):
        gesture_now = "ZOOM_OUT"
    elif detect_stop(hands_data):
        gesture_now = "STOP"

    # Stabilizacja
    if gesture_now == gesture_state["current"]:
        gesture_state["stable_counter"] += 1
    else:
        gesture_state["current"] = gesture_now
        gesture_state["stable_counter"] = 1
        gesture_state["countdown_active"] = False
        gesture_state["countdown"] = 0

    # Odliczanie do aktywacji
    if gesture_now and gesture_state["stable_counter"] >= STABLE_FRAMES:
        if not gesture_state["countdown_active"]:
            gesture_state["countdown_active"] = True
            gesture_state["countdown"] = COUNTDOWN_FRAMES

        if gesture_state["countdown"] > 0:
            seconds_left = gesture_state["countdown"] // FPS
            cv2.putText(frame, f"{gesture_now} in {seconds_left}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            gesture_state["countdown"] -= 1
            if gesture_state["countdown"] == 0:
                gesture_counter[gesture_now] += 1
                gesture_state["last_detected"] = gesture_now
                gesture_state["timer"] = gesture_state["display_frames"]
                print(f"[GESTURE] {gesture_now} triggered")
    else:
        gesture_state["countdown_active"] = False
        gesture_state["countdown"] = 0

    # Wyświetl aktywny gest
    if gesture_state["timer"] > 0:
        cv2.putText(frame, f"Gesture: {gesture_state['last_detected']}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        gesture_state["timer"] -= 1
