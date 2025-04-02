"""
OAK_DS2.md.py: Odpowiada za przechwytywanie obrazu z kamery OAK-D i rozpoznawanie gestów na żywo oraz zarządzanie trybem myszki.
"""
import pyautogui
from mouse_control_functions import *
from gesture_mode_functions import *
import cv2
import mediapipe as mp
import depthai as dai
import pygame
import time

pygame.mixer.init()
click_sound = pygame.mixer.Sound("static/click.mp3")

ACTIVATION_SECONDS = 4.0
TOGGLE_DISPLAY_TIME = 2.0
CURSOR_SMOOTHING = 0.5
CURSOR_PREVIEW_SECONDS = 2.0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# temporary
gesture_counter = {
    "stop": 0,
    "zoom_in:": 1,
    "Zoom_out": 2,
    # inne gesty...
}

def run_mouse_toggle():
    mouse_mode = False
    mouse_mode_start_time = None
    status_message = ""
    status_time = 0
    clicking = False
    last_cursor_pos = None
    cursor_visible = False
    cursor_preview_start_time = None
    grab_display_time = 0

    pipeline = dai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    xout_video = pipeline.createXLinkOut()

    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setVideoSize(1920, 1080)
    cam_rgb.setFps(30)
    cam_rgb.setInterleaved(False)

    xout_video.setStreamName("video")
    cam_rgb.video.link(xout_video.input)

    device = dai.Device(pipeline)
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)

    try:
        with mp_hands.Hands(model_complexity=1, max_num_hands=2,
                            min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
            while True:
                in_frame = video_queue.get()
                if in_frame is None:
                    continue

                frame = in_frame.getCvFrame()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                h, w, _ = frame.shape
                overlay_text = ""
                left_hand = None
                right_hand = None

                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        landmarks = hand_landmarks.landmark
                        label = results.multi_handedness[idx].classification[0].label

                        if label == "Left":
                            left_hand = landmarks
                        else:
                            right_hand = landmarks

                        cx = int(landmarks[0].x * w)
                        cy = int(landmarks[0].y * h)
                        cv2.putText(frame, f"{label}", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # === PIĘŚĆ AKTYWUJE TRYB MYSZY ===
                if left_hand and is_fist(left_hand):
                    cx = left_hand[0].x
                    cy = left_hand[0].y
                    if cx < 0.2 and cy < 0.5:  # Dłoń zaciśnięta i wystawiona w lewo, z dala od głowy
                        if mouse_mode_start_time is None:
                            mouse_mode_start_time = time.time()
                        elapsed = time.time() - mouse_mode_start_time
                        overlay_text = f"Mouse Activation: {elapsed:.1f} / {ACTIVATION_SECONDS}s"
                        if elapsed >= ACTIVATION_SECONDS:
                            mouse_mode = not mouse_mode
                            status_message = "Mouse Mode: ENABLED" if mouse_mode else "Mouse Mode: DISABLED"
                            status_time = time.time()
                            mouse_mode_start_time = None
                            try:
                                click_sound.play()
                            except:
                                pass
                    else:
                        mouse_mode_start_time = None
                        overlay_text = "Move left fist farther from face\nto activate mouse mode"
#######################################################################################################################
                if not mouse_mode:
                    hands_data = []

                    if results.multi_hand_landmarks and results.multi_handedness:
                        for lm, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            hand_type = handness.classification[0].label
                            hands_data.append({
                                "type": hand_type,
                                "landmarks": lm.landmark
                            })
                    run_gesture_mode(frame, hands_data)
                else:
                    # === PALEC WSKAZUJĄCY PRZEJMUJE KURSOR ===
                    if left_hand and is_pointing(left_hand):
                        screen_w, screen_h = pyautogui.size()
                        x = left_hand[8].x
                        y = left_hand[8].y
                        new_x = int(screen_w * x)
                        new_y = int(screen_h * y)
                        if last_cursor_pos:
                            new_x = int(last_cursor_pos[0] * (1 - CURSOR_SMOOTHING) + new_x * CURSOR_SMOOTHING)
                            new_y = int(last_cursor_pos[1] * (1 - CURSOR_SMOOTHING) + new_y * CURSOR_SMOOTHING)
                        pyautogui.moveTo(new_x, new_y)
                        last_cursor_pos = (new_x, new_y)
                        cursor_visible = True
                    else:
                        last_cursor_pos = None  # 👈 dodaj to lub usun.

                    if right_hand:
                        # === ZABLOKOWANIE KLİKANIE Gdy lewa ręka jest w pozycji "STOP" ===
                        if left_hand and is_stop(left_hand):  # Jeśli lewa ręka jest w pozycji "STOP"
                            pass  # Blokujemy akcje prawej ręki
                        else:
                            if is_extended(right_hand):  # Jeśli ręka jest wyprostowana
                                pass  # Zablokuj kliknięcie i grab file (brak akcji)
                            elif is_okay_gesture(right_hand):  # Jeżeli dłoń jest w pozycji wskazującej
                                pyautogui.click()
                                cx = int(right_hand[12].x * w)
                                cy = int(right_hand[12].y * h)
                                cv2.putText(frame, "CLICK", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                time.sleep(0.3)
                            elif is_fist(right_hand) and not clicking:
                                pyautogui.mouseDown()
                                clicking = True
                                grab_display_time = time.time()
                            elif not is_fist(right_hand) and clicking:
                                pyautogui.mouseUp()
                                clicking = False

                color = (0, 200, 0) if mouse_mode else (0, 0, 255)
                text = "MOUSE MODE: ENABLED" if mouse_mode else "GESTURE MODE: ENABLED"
                cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                if status_message and time.time() - status_time <= TOGGLE_DISPLAY_TIME:
                    cv2.putText(frame, status_message, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                if overlay_text:
                    for i, line in enumerate(overlay_text.split('\n')):
                        cv2.putText(frame, line, (10, h - 50 - i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

                if cursor_visible and left_hand:
                    cx = int(left_hand[8].x * w)
                    cy = int(left_hand[8].y * h)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

                if time.time() - grab_display_time <= 0.8 and right_hand:
                    cx = int(right_hand[0].x * w)
                    cy = int(right_hand[0].y * h)
                    cv2.putText(frame, "GRAB FILE", (cx, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                cv2.imshow("Mouse vs Gesture Mode", frame)
                if cv2.waitKey(10) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        print("\n🛑 Zatrzymano program (Ctrl+C)")
    finally:
        print("\n📊 Liczba rozpoznanych gestów:")
        for gesture, count in gesture_counter.items():
            print(f" - {gesture}: {count}")
        cv2.destroyAllWindows()

run_mouse_toggle()
