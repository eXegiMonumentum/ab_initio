# 2️⃣ Przygotowanie danych

# skrypt, który:
# ✅ Ładuje dane z CSV
# ✅ Dodaje etykiety do każdego powtórzenia gestu
# ✅ Standaryzuje długość sekwencji (np. interpolacja do stałej liczby klatek)
# ✅ Przygotowuje dane do modelu ML

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# Ścieżka do katalogu z danymi
output_dir = "gestures_data"


def load_gesture_data(gesture_name, label, target_frames=100):
    """
    Wczytuje dane dla danego gestu, interpoluje do stałej liczby klatek i dodaje etykietę.

    :param gesture_name: Nazwa gestu (folder)
    :param label: Etykieta gestu (np. 0 dla "shake", 1 dla "wave")
    :param target_frames: Docelowa liczba klatek dla każdej próbki
    :return: Lista sekwencji znormalizowanych danych
    """
    gesture_dir = os.path.join(output_dir, gesture_name)
    files = [f for f in os.listdir(gesture_dir) if f.endswith("_positions.csv")]

    data_list = []

    for file in files:
        csv_path = os.path.join(gesture_dir, file)
        data = pd.read_csv(csv_path, names=["frame", "x", "y", "z"])

        # Interpolacja do target_frames klatek
        interp_x = interp1d(np.linspace(0, 1, len(data)), data["x"], kind="linear")
        interp_y = interp1d(np.linspace(0, 1, len(data)), data["y"], kind="linear")
        interp_z = interp1d(np.linspace(0, 1, len(data)), data["z"], kind="linear")

        new_x = interp_x(np.linspace(0, 1, target_frames))
        new_y = interp_y(np.linspace(0, 1, target_frames))
        new_z = interp_z(np.linspace(0, 1, target_frames))

        sequence = np.column_stack([new_x, new_y, new_z])
        data_list.append((sequence, label))  # Dodajemy etykietę

    return data_list


# Przykład użycia - wczytaj gesty i oznacz je etykietami
gestures = {
    "shake": 0,  # Gest "shake" = 0
    "wave": 1,  # Gest "wave" = 1 (jeśli masz drugi gest)
}

dataset = []
for gesture, label in gestures.items():
    dataset.extend(load_gesture_data(gesture, label))

# Konwersja do macierzy NumPy
X = np.array([data[0] for data in dataset])
y = np.array([data[1] for data in dataset])

# Standaryzacja danych (normalizacja)
scaler = StandardScaler()
X = np.array([scaler.fit_transform(seq) for seq in X])

print(f"Przygotowano {len(X)} próbek do nauki.")
