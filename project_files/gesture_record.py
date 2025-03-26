import os
import csv
import cv2
import depthai as dai
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 📂 Katalog bazowy na gesty
output_dir = "gestures_data"
os.makedirs(output_dir, exist_ok=True)

# Ustawienia dla MediaPipe do rozpoznawania dłoni
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 📌 Funkcja do znalezienia numeru pliku dla danego gestu
def get_next_file_number(gesture_name):
    gesture_dir = os.path.join(output_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(gesture_dir) if
                      f.startswith(f"{gesture_name}_") and f.endswith("_positions.csv")]

    if not existing_files:
        return 1  # Jeśli nie ma plików, zaczynamy od 1

    numbers = [int(f.split("_")[1]) for f in existing_files]
    return max(numbers) + 1

# 📡 Funkcja sprawdzająca, czy kamera jest podłączona
def check_camera_connection():
    try:
        # Tworzymy pipeline DepthAI
        pipeline = dai.Pipeline()

        # Tworzymy kamerę RGB
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

        # Ustawiamy rozdzielczość RGB na 640x480 (jeśli chcesz VGA)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_640x480)  # Użycie THE_640x480

        # Uruchamiamy urządzenie i pipeline
        with dai.Device(pipeline) as device:
            # Jeśli urządzenie zostało wykryte, zwrócimy True
            if device is not None:
                return True
            else:
                return False
    except Exception as e:
        print(f"❌ Błąd: {str(e)}")
        return False

# 🎥 Nagrywanie gestów
def record_gesture(gesture_name):
    """Nagrywa gest (RGB + Depth) i zapisuje dane o trajektorii dłoni do nowego pliku CSV."""
    # Sprawdzanie, czy urządzenie DepthAI jest podłączone
    if not check_camera_connection():
        print("❌ Kamera DepthAI nie jest podłączona. Upewnij się, że kamera jest prawidłowo podłączona.")
        return

    # Tworzymy pipeline DepthAI
    pipeline = dai.Pipeline()

    # Tworzymy kamerę RGB
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    # Ustawienie rozdzielczości kamery RGB na 1080p
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080P)  # Użycie THE_1080P
    cam_rgb.setFps(30)

    # Ustawiamy kamerę Depth
    cam_depth = pipeline.create(dai.node.StereoDepth)
    cam_depth.setBoardSocket(dai.CameraBoardSocket.LEFT)

    # Ustawienie rozdzielczości Depth kamery na 720p
    cam_depth.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720P)  # Użycie THE_720P
    cam_depth.setFps(30)
    cam_depth.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Tworzymy outputy do przesyłania obrazów
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("video")
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    cam_rgb.video.link(xout_rgb.input)
    cam_depth.depth.link(xout_depth.input)

    # Uruchamiamy urządzenie i pipeline
    with dai.Device(pipeline) as device:
        # Kolejki do odbierania wideo RGB i głębi
        q_rgb = device.getOutputQueue(name="video", maxSize=8, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

        gesture_dir = os.path.join(output_dir, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)

        file_number = get_next_file_number(gesture_name)
        csv_path = os.path.join(gesture_dir, f"{gesture_name}_{file_number:02d}_positions.csv")

        # Pliki wideo RGB i Depth
        video_rgb_path = os.path.join(gesture_dir, f"{gesture_name}_rgb.avi")
        video_depth_path = os.path.join(gesture_dir, f"{gesture_name}_depth.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_rgb = cv2.VideoWriter(video_rgb_path, fourcc, 20.0, (1920, 1080))
        out_depth = cv2.VideoWriter(video_depth_path, fourcc, 20.0, (512, 424))

        frame_count = 0
        print(f"🔴 Nagrywanie gestu '{gesture_name}'... Wciśnij 'q' aby zakończyć.")

        while True:
            # Odbieranie klatek RGB i Depth
            frame_rgb = q_rgb.get()
            frame_depth = q_depth.get()

            rgb_image = frame_rgb.getCvFrame()
            depth_image = frame_depth.getCvFrame()

            # Zapisz wideo RGB i Depth
            out_rgb.write(rgb_image)
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            out_depth.write(depth_colored)

            # Przygotowanie do zapisania danych o trajektorii dłoni
            rgb_frame_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame_rgb)

            if results.multi_hand_landmarks:
                with open(csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    for landmarks in results.multi_hand_landmarks:
                        for landmark in landmarks.landmark:
                            writer.writerow([frame_count, landmark.x, landmark.y, landmark.z])
                        mp_drawing.draw_landmarks(rgb_image, landmarks, mp_hands.HAND_CONNECTIONS)

            # Pokaż wideo RGB i Depth
            cv2.imshow("RGB Video", rgb_image)
            cv2.imshow("Depth Video", depth_colored)

            frame_count += 1

            # Sprawdzenie, czy użytkownik chce zakończyć nagrywanie
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('RGB Video', cv2.WND_PROP_VISIBLE) < 1:
                print("✅ Nagrywanie zakończone!")
                break

        out_rgb.release()
        out_depth.release()
        cv2.destroyAllWindows()

# 🎞 Animacja gestu
def animate_gesture(gesture_name):
    """Tworzy animację dla podanego gestu."""
    gesture_dir = os.path.join(output_dir, gesture_name)
    csv_files = [f for f in os.listdir(gesture_dir) if f.endswith("_positions.csv")]

    if not csv_files:
        print(f"❌ Brak danych dla gestu '{gesture_name}'.")
        return

    print("📂 Dostępne pliki CSV:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")

    choice = int(input("Wybierz numer pliku do animacji: ")) - 1
    if choice < 0 or choice >= len(csv_files):
        print("❌ Niepoprawny wybór.")
        return

    csv_path = os.path.join(gesture_dir, csv_files[choice])
    data = pd.read_csv(csv_path, names=["frame", "x", "y", "z"])

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(f"Animacja gestu: {gesture_name}")

    scatter, = ax.plot([], [], "ro", markersize=5)

    def update(frame):
        current_data = data[data["frame"] == frame]
        scatter.set_data(current_data["x"], current_data["y"])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=int(data["frame"].max()), interval=50, blit=True)
    plt.show()

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
    print("""🎮 Wybierz tryb:
    1. 🔴 Nagrywanie nowego gestu
    2. 🎥 Animacja istniejącego gestu
    3. 📊 Wizualizacja trajektorii dłoni w 3D
    4. ❌ Wyjście""")
    choice = input("Wybierz opcję: ")
    if choice == "1":
        gesture_name = input("Podaj nazwę gestu: ")
        record_gesture(gesture_name)
    elif choice == "2":
        gesture_name = input("Podaj nazwę gestu do animacji: ")
        animate_gesture(gesture_name)
    elif choice == "3":
        gesture_name = input("Podaj nazwę gestu do wizualizacji: ")
        plot_finger_positions_3D(gesture_name)
    elif choice == "4":
        exit()

if __name__ == "__main__":
    while True:
        main()
