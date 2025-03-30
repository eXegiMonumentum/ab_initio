
"""
1️⃣ Nagrywanie danych (gesture_record.py)
2️⃣ Przygotowanie danych (prepare_gesture_data.py)
3️⃣ Trening modelu (train_gesture_lstm.py)
4️⃣ Ewaluacja modelu (evaluate_model.py)
5️⃣ 🔜 Rozpoznawanie gestów na żywo z kamery (predict_live.py)
"""

from gesture_record_help import *
import csv
import cv2
import depthai as dai
import mediapipe as mp
import time


def record_gesture(gesture_name, max_time=10, mirror_flip=True):
    """ MediaPipe śledzi rękę w postaci 21 punktów 3D – każdy palec i nadgarstek.
        Dzięki temu różny rozstaw palców czy ruch będzie odwzorowany w danych (CSV).
        Twoje dane będą tym lepsze, im więcej „wariantów” danego gestu nagrasz – czyli np.:
        „shake” z różną energią
        „shake” z szeroko rozstawionymi palcami
        „shake” wykonany przez różne osoby

        !!!
            Dla każdego gestu warto mieć nagrania:

            z 3+ różnych kątów (np. na wprost, z boku, od góry),

            w 2-3 tempach (powolny, normalny, szybki),

            z małą losowością (nie musi być zawsze idealny ruch),

            lewą i prawą ręką (jeśli chcesz obsługiwać obie).
        !!!
    """

    # Główny katalog na dane
    output_dir = "gestures_data"
    os.makedirs(output_dir, exist_ok=True)

    # Inicjalizacja MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Tworzenie pipeline'u DepthAI
    pipeline = dai.Pipeline()

    # Konfiguracja kamery RGB
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)
    cam_rgb.setInterleaved(False)

    # Kamery mono do głębi
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # Konfiguracja stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Łączenie kamer z modułem stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Strumienie wyjściowe
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("video")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="video", maxSize=8, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

        gesture_dir = os.path.join(output_dir, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)

        # Generowanie nowego numeru pliku CSV
        def get_next_file_number(gesture_name):
            gesture_dir = os.path.join("gestures_data", gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)
            existing_files = [f for f in os.listdir(gesture_dir)
                              if f.startswith(f"{gesture_name}_") and f.endswith("_positions.csv")]
            numbers = []
            for f in existing_files:
                try:
                    num_str = f[len(gesture_name) + 1:].split("_")[0]
                    number = int(num_str)
                    numbers.append(number)
                except:
                    continue
            return max(numbers) + 1 if numbers else 1

        file_number = get_next_file_number(gesture_name)
        csv_path = os.path.join(gesture_dir, f"{gesture_name}_{file_number:02d}_positions.csv")

        # Pobierz pierwsze klatki dla konfiguracji VideoWriterów
        frame_rgb = q_rgb.get().getCvFrame()
        frame_depth = q_depth.get().getCvFrame()

        video_rgb_path = os.path.join(gesture_dir, f"{gesture_name}_rgb.avi")
        video_depth_path = os.path.join(gesture_dir, f"{gesture_name}_depth.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_rgb = cv2.VideoWriter(video_rgb_path, fourcc, 20.0, (frame_rgb.shape[1], frame_rgb.shape[0]))
        out_depth = cv2.VideoWriter(video_depth_path, fourcc, 20.0, (frame_depth.shape[1], frame_depth.shape[0]))

        print(f"🔴 Nagrywanie gestu '{gesture_name}'... Wciśnij 'q' aby zakończyć.")

        # Tworzenie nagłówka pliku CSV
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "x", "y", "z", "hand"])

        frame_count = 0
        start_time = time.time()

        while True:
            # Pobranie klatek
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            rgb_image = in_rgb.getCvFrame()
            if mirror_flip:
                rgb_image = cv2.flip(rgb_image, 1)  # Odbicie lustrzane obrazu

            depth_image = in_depth.getFrame()
            rgb_input = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_input)

            # Przetwarzanie wykrytych dłoni
            if results.multi_hand_landmarks and results.multi_handedness:
                with open(csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    for idx, landmarks in enumerate(results.multi_hand_landmarks):
                        hand_label = results.multi_handedness[idx].classification[0].label

                        # Pozycja nadgarstka (do podpisu)
                        h, w, _ = rgb_image.shape
                        cx = int(landmarks.landmark[0].x * w)
                        cy = int(landmarks.landmark[0].y * h)

                        # Heurystyka – określenie ręki na podstawie pozycji kciuka
                        thumb_tip = landmarks.landmark[4]
                        index_tip = landmarks.landmark[8]
                        hand_side = "Right?" if thumb_tip.x < index_tip.x else "Left?"

                        # Dodanie etykiet do obrazu
                        cv2.putText(rgb_image, f"MP: {hand_label}", (cx, cy - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(rgb_image, f"Heur: {hand_side}", (cx, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        # Rysowanie linii dłoni
                        mp_drawing.draw_landmarks(rgb_image, landmarks, mp_hands.HAND_CONNECTIONS)

                        # Zapis pozycji punktów do CSV
                        for landmark in landmarks.landmark:
                            writer.writerow([frame_count, landmark.x, landmark.y, landmark.z, hand_label])

            # Zapis wideo RGB z overlayami
            out_rgb.write(rgb_image)

            # Przetworzenie i zapis głębi
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            out_depth.write(depth_colored)

            # Wyświetlanie podglądu
            cv2.imshow("🟢 RGB Video (kamera)", rgb_image)
            cv2.imshow("🔵 Depth Video (głębia)", depth_colored)
            cv2.setWindowProperty("🟢 RGB Video (kamera)", cv2.WND_PROP_TOPMOST, 1)

            # Sprawdzenie zakończenia
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("✅ Nagrywanie zakończone!")
                break

            if time.time() - start_time >= max_time:
                print("⏹️ Czas nagrania upłynął.")
                break

        # Zwolnienie zasobów
        out_rgb.release()
        out_depth.release()
        cv2.destroyAllWindows()


# 🏆 Główne menu użytkownika
def main():
    print("👋 Witaj w systemie nagrywania i analizy gestów")
    gesture_name = input("📁 Podaj nazwę gestu (np. shake, wave): ").strip()

    while True:
        print(f"""\n🎮 Wybierz tryb dla gestu '{gesture_name}':
        1. 🔴 Nagrywanie nowego gestu (RGB + Depth + Pozycje dłoni)
        2. 🎥 Animacja istniejącego gestu
        3. 📊 Wizualizacja trajektorii dłoni w 3D
        4. 🏁 Zmień nazwę gestu
        5. ❌ Wyjście
        6. 🚀 Rozpoznawanie gestów na żywo
        0. zip_gesture_data
        """)

        choice = input("Wybierz opcję: ").strip()

        if choice == "1":
            record_gesture(gesture_name, max_time=4)
        elif choice == "2":
            animate_gesture(gesture_name)
        elif choice == "3":
            plot_finger_positions_3D(gesture_name)
        elif choice == "4":
            new_name = input("🔁 Podaj nową nazwę gestu: ").strip()
            rename_gesture(gesture_name, new_name)
        elif choice == "5":
            print("👋 Zakończono program.")
            break

        elif choice == "6":
            from predict_live import run_live_prediction
            run_live_prediction()

        elif choice == "0":
            from gesture_record_help import zip_gesture_data_snapshot
            zip_gesture_data_snapshot()

        else:
            print("❌ Niepoprawny wybór. Spróbuj ponownie.")


if __name__ == "__main__":
    while True:
        main()

# mogę dodać:
# 📦 Możliwość wyboru liczby klatek, fps lub rozdzielczości przez użytkownika


# 📁 Dane trafiają do:
# gestures_data/<nazwa_gestu>/<nazwa_gestu>_01_positions.csv
# 🟢 Przykład gestu: "shake" – potrząsanie dłonią

# Następuje przygotowanie danych do modelu LSTM [2].

# [1] gesture_record.py  →  [2] prepare_gesture_data.py  →  [3] train_gesture_lstm.py  →  [4] evaluate_model.py  →  [5] predict_live.py