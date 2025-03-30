"""
prepare_gesture_data.py

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
gestures = {
    "shake": 0,
    "wave": 1,
    # dodaj więcej gestów tutaj
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



# ✅ 2. Przygotowanie danych do uczenia
# 📁 Plik: prepare_gesture_data.py
# 🎯 Co robi:

# Wczytuje wszystkie pliki CSV z gestures_data/
# Grupuje po klatkach i wybiera jedną dłoń
# Interpoluje dane do tej samej długości (target_frames)
# Normalizuje dane

# Zapisuje do:
# X_gestures.npy – dane wejściowe (sekwencje)
# y_gestures.npy – etykiety (klasy gestów)