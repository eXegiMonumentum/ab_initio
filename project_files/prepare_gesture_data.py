"""
1️⃣ Nagrywanie danych (gesture_record.py)
2️⃣ Przygotowanie danych (prepare_gesture_data.py)
3️⃣ Trening modelu (train_gesture_lstm.py)
4️⃣ Ewaluacja modelu (evaluate_model.py)
5️⃣ Rozpoznawanie gestów na żywo z kamery (predict_live.py)
"""

"""
Skrypt wczytuje pliki CSV z folderu gestures_data, przetwarza je do postaci
macierzy NumPy gotowych do trenowania modelu uczenia maszynowego (np. LSTM).

Każdy gest jest reprezentowany jako sekwencja 21 punktów 3D (x, y, z) na każdą klatkę.
Skrypt interpoluje każdą sekwencję do stałej liczby klatek (np. 100) i normalizuje dane.

Zwraca dwie tablice NumPy:
- X: dane wejściowe, kształt (num_samples, target_frames, 63)
- y: etykiety gestów (0, 1, ...)

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# Główny katalog z danymi
output_dir = "gestures_data"

def load_gesture_data(gesture_name, label, target_frames=100):
    """
    Wczytuje i przetwarza dane z plików CSV danego gestu.
    Każdy plik jest interpolowany do stałej liczby klatek i normalizowany.

    Zwraca: lista (sequence, label)
    """
    gesture_dir = os.path.join(output_dir, gesture_name)
    files = [f for f in os.listdir(gesture_dir) if f.endswith("_positions.csv")]

    data_list = []

    for file in files:
        csv_path = os.path.join(gesture_dir, file)
        df = pd.read_csv(csv_path, names=["frame", "x", "y", "z", "hand"])

        # Grupowanie po klatkach i wybieranie tylko jednej dłoni (najczęściej prawej)
        hand_to_use = df["hand"].mode()[0]  # najczęstsza wartość
        df = df[df["hand"] == hand_to_use]  # filtruj tylko wybraną dłoń

        frames = []
        for frame_id, group in df.groupby("frame"):
            coords = group[["x", "y", "z"]].values.flatten()
            if len(coords) == 63:
                frames.append(coords)

        if len(frames) < 10:
            continue  # pomiń zbyt krótkie sekwencje

        frames = np.array(frames)

        # Interpolacja do jednakowej liczby klatek
        interp_frames = []
        for i in range(frames.shape[1]):
            interp_func = interp1d(np.linspace(0, 1, len(frames)), frames[:, i], kind="linear")
            interp_col = interp_func(np.linspace(0, 1, target_frames))
            interp_frames.append(interp_col)

        interp_sequence = np.stack(interp_frames, axis=1)  # shape = (target_frames, 63)
        data_list.append((interp_sequence, label))

    return data_list

# === Definicja etykiet gestów ===
if __name__ == "__main__":
    gestures = {
        "shake": 0,
        "wave": 1,
        "stop": 2,
        "wave_both": 3,
        "freeze": 4,
        "wiggle_V": 5,
        "shake_sidechains": 6,
        "rebuild": 7,
        "zoom_in": 8,
        "zoom_out": 9,
    }

    dataset = []
    for gesture, label in gestures.items():
        dataset.extend(load_gesture_data(gesture, label))

    # === Przygotowanie danych ===
    X = np.array([seq for seq, _ in dataset])
    y = np.array([lbl for _, lbl in dataset])

    # === Normalizacja danych ===
    scaler = StandardScaler()
    X = np.array([scaler.fit_transform(seq) for seq in X])

    print(f"✅ Przygotowano {len(X)} przykładów, shape: {X.shape}, etykiety: {set(y)}")


    # === Zapisz dane do plików .npy ===
    np.save("X_gestures.npy", X)
    np.save("y_gestures.npy", y)
    print("💾 Dane zapisane jako X_gestures.npy i y_gestures.npy")

    # === Animowana wizualizacja 3D całej dłoni ===
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    sample = X[0]  # Pierwsza sekwencja (target_frames, 63)

    # Przygotuj dane jako 21 punktów (x, y, z) na każdą klatkę
    points_per_frame = sample.reshape(-1, 21, 3)  # shape = (target_frames, 21, 3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c='red', s=40)

    # Ustawienia osi
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("🖐 Animacja 3D trajektorii dłoni")

    def update(frame_idx):
        coords = points_per_frame[frame_idx]
        sc._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        return sc,

    ani = FuncAnimation(fig, update, frames=len(points_per_frame), interval=100, blit=False)
    plt.show()



# 2. Przygotowanie danych – prepare_gesture_data.py
# Wczytuje dane CSV z wielu plików i gestów.
# Dla każdej klatki tworzy 63-wymiarowy wektor: 21 punktów × 3 współrzędne.
# Interpoluje każdą próbkę do tej samej długości (np. 100 klatek).
# Normalizuje dane (standaryzacja).

# Zapisuje dane do .npy:
# X_gestures.npy, y_gestures.npy

# dane .npy przyjmuje plik train_gesture_lstm.py
