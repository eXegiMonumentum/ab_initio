import os
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Sprawdzenie dostępności Kinecta i MediaPipe
try:
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime

    kinect = PyKinectRuntime.PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    print("✅ Kinect działa poprawnie!")
    kinect_available = True
except ImportError as e:
    print(f"❌ Błąd importu pykinect2: {e}")
    kinect_available = False
except Exception as e:
    print(f"❌ Inny błąd: {e}")
    kinect_available = False

try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
    mp_drawing = mp.solutions.drawing_utils
    mediapipe_available = True
except ImportError:
    mediapipe_available = False

# 📂 Katalog bazowy na gesty
output_dir = "gestures_data"
os.makedirs(output_dir, exist_ok=True)


# 📌 Funkcja do znalezienia numeru pliku dla danego gestu
def get_next_file_number(gesture_name):
    """Znajduje następny numer pliku dla danego gestu."""
    gesture_dir = os.path.join(output_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(gesture_dir) if
                      f.startswith(f"{gesture_name}_") and f.endswith("_positions.csv")]

    if not existing_files:
        return 1

    numbers = [int(f.split("_")[1]) for f in existing_files]
    return max(numbers) + 1


# 🎥 Nagrywanie gestów
def record_gesture(gesture_name):
    """Nagrywa gest (RGB + Depth) i zapisuje dane o trajektorii dłoni do nowego pliku CSV."""
    if not kinect_available or not mediapipe_available:
        print("❌ Kinect lub MediaPipe nie są dostępne! Nie można nagrywać.")
        return

    gesture_dir = os.path.join(output_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    file_number = get_next_file_number(gesture_name)
    csv_path = os.path.join(gesture_dir, f"{gesture_name}_{file_number:02d}_positions.csv")

    video_rgb_path = os.path.join(gesture_dir, f"{gesture_name}_rgb.avi")
    video_depth_path = os.path.join(gesture_dir, f"{gesture_name}_depth.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_rgb = cv2.VideoWriter(video_rgb_path, fourcc, 20.0, (1920, 1080))
    out_depth = cv2.VideoWriter(video_depth_path, fourcc, 20.0, (512, 424))

    frame_count = 0
    print(f"🔴 Nagrywanie... Plik: {csv_path} | Wciśnij 'q' aby zakończyć.")

    while True:
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            rgb_image = color_frame.reshape((1080, 1920, 4))[:, :, :3]
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
            out_rgb.write(rgb_image)

        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_image = depth_frame.reshape((424, 512))
            depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            out_depth.write(depth_colored)

        rgb_frame_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame_rgb)

        if results.multi_hand_landmarks:
            with open(csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                for landmarks in results.multi_hand_landmarks:
                    for landmark in landmarks.landmark:
                        writer.writerow([frame_count, landmark.x, landmark.y, landmark.z])
                    mp_drawing.draw_landmarks(rgb_image, landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("RGB Video", rgb_image)
        cv2.imshow("Depth Video", depth_colored)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_rgb.release()
    out_depth.release()
    cv2.destroyAllWindows()
    print("✅ Nagrywanie zakończone!")


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


# 📊 Wizualizacja 3D
from mpl_toolkits.mplot3d import Axes3D


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
    # Z → góra/dół (Vertical) - odwrócona
    ax.set_xlabel("X Position (Lateral - Left/Right)")
    ax.set_ylabel("Z Position (Vertical - Up/Down)")
    ax.set_zlabel("Y Position (Depth - Forward/Backward)")
    ax.set_title(f"Animacja trajektorii dłoni: {gesture_name}")

    # Odwrócenie osi Z (góra/dół), aby machanie było do góry
    ax.invert_yaxis()  # 🔄 Odwrócenie osi Y (która reprezentuje Z!)

    # Początkowy punkt trajektorii
    line, = ax.plot([], [], [], 'r-', lw=2)  # Czerwona linia

    def update(frame):
        """Aktualizuje wykres dla danej klatki."""
        current_data = data[data["frame"] <= frame]
        line.set_data(current_data["x"], current_data["z"])  # X i Z zamienione
        line.set_3d_properties(current_data["y"])  # Y w miejsce głębokości
        return line,

    ani = animation.FuncAnimation(fig, update, frames=int(data["frame"].max()), interval=50, blit=False)
    plt.show()


# 🏆 Główne menu użytkownika
def main():
    print("""
🎮 Wybierz tryb:
1. 🔴 Nagrywanie nowego gestu
2. 🎥 Animacja istniejącego gestu
3. 📊 Wizualizacja trajektorii dłoni w 3D
4. ❌ Wyjście
""")
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
