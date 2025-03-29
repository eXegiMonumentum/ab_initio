import os
import csv
import cv2
import depthai as dai
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import time

def record_gesture(gesture_name, max_time=10):
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

    output_dir = "gestures_data"
    os.makedirs(output_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    pipeline = dai.Pipeline()

    # === Kamera RGB ===
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)
    cam_rgb.setInterleaved(False)

    # === Kamery Mono do StereoDepth ===
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # === StereoDepth ===
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # === Połączenia ===
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # === Outputy ===
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

        # Znajdź kolejny numer pliku
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

        # Poczekaj na pierwsze klatki
        frame_rgb = q_rgb.get().getCvFrame()
        frame_depth = q_depth.get().getCvFrame()

        video_rgb_path = os.path.join(gesture_dir, f"{gesture_name}_rgb.avi")
        video_depth_path = os.path.join(gesture_dir, f"{gesture_name}_depth.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_rgb = cv2.VideoWriter(video_rgb_path, fourcc, 20.0, (frame_rgb.shape[1], frame_rgb.shape[0]))
        out_depth = cv2.VideoWriter(video_depth_path, fourcc, 20.0, (frame_depth.shape[1], frame_depth.shape[0]))

        print(f"🔴 Nagrywanie gestu '{gesture_name}'... Wciśnij 'q' aby zakończyć.")

        # Utwórz plik CSV z nagłówkiem
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "x", "y", "z", "hand"])

        frame_count = 0
        start_time = time.time()

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            rgb_image = in_rgb.getCvFrame()
            depth_image = in_depth.getFrame()

            out_rgb.write(rgb_image)
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            out_depth.write(depth_colored)

            rgb_input = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_input)

            if results.multi_hand_landmarks and results.multi_handedness:
                with open(csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    for idx, landmarks in enumerate(results.multi_hand_landmarks):
                        hand_label = results.multi_handedness[idx].classification[0].label
                        for landmark in landmarks.landmark:
                            writer.writerow([frame_count, landmark.x, landmark.y, landmark.z, hand_label])
                        mp_drawing.draw_landmarks(rgb_image, landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("RGB Video", rgb_image)
            cv2.imshow("Depth Video", depth_colored)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("✅ Nagrywanie zakończone!")
                break

            if time.time() - start_time >= max_time:
                print("⏹️ Czas nagrania upłynął.")
                break

        out_rgb.release()
        out_depth.release()
        cv2.destroyAllWindows()


# 📊 Wizualizacja 3D trajektorii dłoni
def plot_finger_positions_3D(gesture_name):
    """Tworzy animację 3D trajektorii dłoni na podstawie CSV, dostosowaną do naturalnej orientacji człowieka."""
    csv_path = os.path.join(output_dir, gesture_name, f"{gesture_name}_positions.csv")

    if not os.path.exists(csv_path):
        print("❌ Brak danych do wizualizacji.")
        return

    data = pd.read_csv(csv_path, names=['frame', 'x', 'y', 'z'])
    if data.empty:
        print("⚠ Plik CSV jest pusty.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Zmiana kolejności osi:
    # X → lewo/prawo (Lateral)
    # Y → przód/tył (Depth)
    # Z → góra/dół (Vertical)
    ax.set_xlabel("X Position (Lateral - Left/Right)")
    ax.set_ylabel("Z Position (Vertical - Up/Down)")
    ax.set_zlabel("Y Position (Depth - Forward/Backward)")
    ax.set_title(f"Animacja trajektorii dłoni: {gesture_name}")

    # Odwrócenie osi Y (góra/dół), aby machanie było do góry
    ax.invert_yaxis()

    # Początkowy punkt trajektorii
    line, = ax.plot([], [], [], 'r-', lw=2)  # Czerwona linia

    def update(frame):
        """Aktualizuje wykres dla danej klatki."""
        current_data = data[data["frame"] <= frame]
        line.set_data(current_data["x"], current_data["y"])  # X i Z zamienione
        line.set_3d_properties(current_data["z"])  # Z w miejsce głębokości
        return line,

    ani = animation.FuncAnimation(fig, update, frames=int(data["frame"].max()), interval=50, blit=False)
    plt.show()

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
        """)
        choice = input("Wybierz opcję: ").strip()

        if choice == "1":
            record_gesture(gesture_name, max_time=3)
        elif choice == "2":
            animate_gesture(gesture_name)
        elif choice == "3":
            plot_finger_positions_3D(gesture_name)
        elif choice == "4":
            gesture_name = input("🔁 Podaj nową nazwę gestu: ").strip()
        elif choice == "5":
            print("👋 Zakończono program.")
            break
        else:
            print("❌ Niepoprawny wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    while True:
        main()


# #
# Co można jeszcze ulepszyć w przyszłości (opcjonalnie):
# 🧠 Rozróżnienie lewej i prawej dłoni (jeśli chcesz rozpoznawać gesty obiema rękami),
#
# 📥 Automatyczna konwersja CSV → .npy po nagraniu,
#
# 🛑 Dodanie limitu nagrania (np. 10 sek), żeby nie nagrać za dużo przypadkowo,
#
# 📦 Możliwość wyboru liczby klatek, fps lub rozdzielczości przez użytkownika