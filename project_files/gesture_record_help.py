"""
przechowuje funkcje dla 1️⃣gesture_record.py)
"""
import os
import shutil
import matplotlib.animation as animation
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import zipfile
import os

import zipfile
import os

def zip_gesture_data_snapshot(output_path="gestures_snapshot.zip", data_dir="gestures_data"):
    """
    Pakuje cały folder gestures_data do pliku ZIP.
    Przydatne np. przed wrzuceniem danych na GitHub.
    """
    os.makedirs("compressed_data", exist_ok=True)
    zip_path = os.path.join("compressed_data", output_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=data_dir)
                zipf.write(full_path, arcname=rel_path)

    print(f"✅ Spakowano {data_dir} → {zip_path}")




def rename_gesture(old_name, new_name):
    base_dir = "gestures_data"
    old_path = os.path.join(base_dir, old_name)
    new_path = os.path.join(base_dir, new_name)

    if not os.path.exists(old_path):
        print(f"❌ Gest '{old_name}' nie istnieje.")
        return
    if os.path.exists(new_path):
        print(f"⚠ Gest docelowy '{new_name}' już istnieje. Wybierz inną nazwę.")
        return

    try:
        shutil.move(old_path, new_path)
        print(f"📁 Folder '{old_name}' został przemianowany na '{new_name}'.")

        # Zmień też nazwy plików w środku
        for filename in os.listdir(new_path):
            old_file_path = os.path.join(new_path, filename)
            new_filename = filename.replace(old_name, new_name)
            new_file_path = os.path.join(new_path, new_filename)
            os.rename(old_file_path, new_file_path)

        print(f"✅ Wszystkie pliki zostały przemianowane.")
    except Exception as e:
        print(f"❌ Błąd podczas zmiany nazwy: {e}")


def animate_gesture(gesture_name):
    gesture_dir = os.path.join("gestures_data", gesture_name)

    # Szukamy najnowszego pliku CSV
    csv_files = sorted([f for f in os.listdir(gesture_dir) if f.endswith("_positions.csv")])
    if not csv_files:
        print("❌ Brak plików CSV dla tego gestu.")
        return

    csv_path = os.path.join(gesture_dir, csv_files[-1])  # najnowszy
    df = pd.read_csv(csv_path)

    # Grupowanie po klatkach
    grouped = df.groupby("frame")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])

    # Ustaw zakresy (opcjonalnie: na podstawie danych)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.2, 0.2)
    ax.view_init(elev=20., azim=-90)

    def update(frame):
        ax.clear()
        frame_data = grouped.get_group(frame)
        xs = frame_data['x'].values
        ys = frame_data['y'].values
        zs = frame_data['z'].values
        hand = frame_data['hand'].values[0]

        color = 'blue' if hand == 'Left' else 'green'
        ax.scatter(xs, ys, zs, c=color, s=50)

        ax.set_title(f"Frame {frame} | Hand: {hand}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-0.2, 0.2)
        ax.view_init(elev=20., azim=-90)

    ani = FuncAnimation(fig, update, frames=grouped.groups.keys(), interval=100)
    plt.show()


# 📊 Wizualizacja 3D trajektorii dłoni
def plot_finger_positions_3D(gesture_name):
    """Tworzy animację 3D trajektorii dłoni na podstawie CSV, dostosowaną do naturalnej orientacji człowieka."""

    gesture_dir = os.path.join("gestures_data", gesture_name)

    # Znajdź najnowszy plik CSV
    csv_files = sorted([f for f in os.listdir(gesture_dir) if f.endswith("_positions.csv")])
    if not csv_files:
        print("❌ Brak danych do wizualizacji.")
        return

    csv_path = os.path.join(gesture_dir, csv_files[-1])  # Najnowszy plik

    data = pd.read_csv(csv_path)
    if data.empty:
        print("⚠ Plik CSV jest pusty.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("X Position (Lateral - Left/Right)")
    ax.set_ylabel("Z Position (Vertical - Up/Down)")
    ax.set_zlabel("Y Position (Depth - Forward/Backward)")
    ax.set_title(f"Animacja trajektorii dłoni: {gesture_name}")
    ax.invert_yaxis()

    line, = ax.plot([], [], [], 'r-', lw=2)

    # Grupowanie po klatkach
    grouped = data.groupby("frame")
    frames = list(grouped.groups.keys())

    def update(frame):
        frame_data = grouped.get_group(frame)
        xs = frame_data['x'].values
        ys = frame_data['y'].values
        zs = frame_data['z'].values
        line.set_data(xs, zs)  # X vs Z (góra/dół)
        line.set_3d_properties(ys)  # Y (głębia)
        return line,

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.show()

    def update(frame):
        """Aktualizuje wykres dla danej klatki."""
        current_data = data[data["frame"] <= frame]
        line.set_data(current_data["x"], current_data["y"])  # X i Z zamienione
        line.set_3d_properties(current_data["z"])  # Z w miejsce głębokości
        return line,

    ani = animation.FuncAnimation(fig, update, frames=int(data["frame"].max()), interval=50, blit=False)
    plt.show()


