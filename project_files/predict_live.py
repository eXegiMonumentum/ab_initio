
"""
1️⃣ Nagrywanie danych (gesture_record.py)
2️⃣ Przygotowanie danych (prepare_gesture_data.py)
3️⃣ Trening modelu (train_gesture_lstm.py)
4️⃣ Ewaluacja modelu (evaluate_model.py)
5️⃣ 🔜 (opcjonalnie) Rozpoznawanie gestów na żywo z kamery (predict_live.py)

🛡️ [Nowość] System został rozszerzony o zabezpieczenia przed niekontrolowanym wywoływaniem skrótów:

- ✋ Minimalna liczba stabilnych rozpoznań (np. 5x z rzędu ten sam gest), zanim nastąpi akcja.
- 🕒 Cooldown czasowy – ogranicza częstotliwość akcji (np. 1 akcja co 2 sekundy).
- 🔁 Pomijanie powtarzających się predykcji tego samego gestu.
- ✅ Użycie `gesture_control.py` do przypisania gestów do skrótów klawiszowych.

System analizuje w czasie rzeczywistym obraz z kamery OAK-D, przetwarza go przez MediaPipe,
zbiera sekwencję 3D punktów dłoni (21 landmarków x [x, y, z]) i klasyfikuje ją za pomocą modelu LSTM.

Wymagania:
- OAK-D + DepthAI
- MediaPipe
- PyTorch
- pyautogui (do obsługi skrótów systemowych)

"""

import depthai as dai
import cv2
import mediapipe as mp
import torch
import numpy as np
import time
from collections import deque
from train_gesture_lstm import GestureLSTM
from gesture_control import handle_gesture


# === DEBUG: logowanie gestów do pliku ===
ENABLE_LOG = True  # ← ustaw na False, aby wyłączyć logi
LOG_FILE = "gesture_log.txt"

def log_event(message):
    if ENABLE_LOG:
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp} {message}\n")


# === Parametry ===
sequence_length = 100
model_path = "gesture_model.pt"
label_map = {0: "shake", 1: "wave", 2: "ok", 3: "thumbs_up"}  # <- zaktualizuj według potrzeb

# === Zabezpieczenia przed spamem ===
last_label = None
last_time = 0
cooldown_time = 2  # sekundy
min_stable_count = 5
stable_counter = 0
                    log_event(f"Nowy gest wykryty: {label}")

# === Inicjalizacja modelu ===
model = GestureLSTM()
try:
    model.load_state_dict(torch.load(model_path))
except FileNotFoundError:
    print(f"❌ Nie znaleziono modelu: {model_path}")
    exit(1)
model.eval()

# === Bufor sekwencji ===
sequence = deque(maxlen=sequence_length)

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === OAK-D Pipeline ===
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")

cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
cam_rgb.setFps(30)
cam_rgb.video.link(xout.input)


def run_live_prediction():
    global last_label, last_time, stable_counter

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("video", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            rgb_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_input)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                coords = []
                for lm in landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                sequence.append(coords)

                if len(sequence) == sequence_length:
                    input_tensor = torch.tensor([sequence], dtype=torch.float32)
                    with torch.no_grad():
                        output = model(input_tensor)
                        prediction = torch.argmax(output, dim=1).item()
                        label = label_map.get(prediction, "Unknown")

                    # Stabilność + cooldown
                    current_time = time.time()
                    if label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 0
                    log_event(f"Nowy gest wykryty: {label}")

                    if stable_counter >= min_stable_count and (current_time - last_time) > cooldown_time:
                        handle_gesture(label)
                        log_event(f"Rozpoznano gest: {label} → wykonano akcję")
                        last_time = current_time

                    last_label = label

                    # Wyświetl nazwę gestu
                    cv2.putText(frame, f"Gest: {label}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Frames: {len(sequence)}/{sequence_length}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("🟢 Rozpoznawanie gestów", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
