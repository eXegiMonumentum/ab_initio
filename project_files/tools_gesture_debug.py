
import os
import csv
import json
import numpy as np
from train_gesture_lstm import GestureLSTM
import torch
from sklearn.metrics import classification_report
from prepare_gesture_data import output_dir

# === Wczytaj model ===
MODEL_PATH = "gesture_model.pt"
model = GestureLSTM()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Mapowanie gestów ===
label_map = {0: "shake", 1: "wave", 2: "ok", 3: "thumbs_up"}
reverse_label_map = {v: k for k, v in label_map.items()}


def test_sequence_from_csv(csv_file_path):
    """Wczytaj sekwencję z CSV i przetestuj, co przewiduje model"""
    df = []
    with open(csv_file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            df.append([float(row['x']), float(row['y']), float(row['z'])])
    
    df = np.array(df).reshape(-1, 21, 3)
    if len(df) < 10:
        print("❌ Za mało danych w sekwencji.")
        return
    
    # Interpolacja do 100 klatek
    from scipy.interpolate import interp1d
    frames = df.reshape(len(df), -1)
    interp_frames = []
    for i in range(frames.shape[1]):
        interp_func = interp1d(np.linspace(0, 1, len(frames)), frames[:, i], kind="linear")
        interp_col = interp_func(np.linspace(0, 1, 100))
        interp_frames.append(interp_col)
    interp_sequence = np.stack(interp_frames, axis=1)  # shape (100, 63)

    # Normalizacja
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_sequence = scaler.fit_transform(interp_sequence)

    # Predykcja
    input_tensor = torch.tensor([norm_sequence], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][prediction].item()

    print(f"🧪 Sekwencja {os.path.basename(csv_file_path)} → {label_map.get(prediction)} ({confidence:.2f})")


def analyze_accuracy_by_gesture():
    """Policz skuteczność rozpoznania osobno dla każdego gestu"""
    X = np.load("X_gestures.npy")
    y = np.load("y_gestures.npy")
    input_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()

    print("🎯 Raport skuteczności:")
    print(classification_report(y, predictions, target_names=label_map.values()))


def log_gesture_to_json(csv_path, label):
    """Zamień CSV z gestem na zapis JSON (np. do analizy lub bazy)"""
    df = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            df.append({
                "frame": int(row["frame"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "hand": row["hand"]
            })

    json_data = {
        "label": label,
        "data": df
    }

    json_name = os.path.splitext(os.path.basename(csv_path))[0] + ".json"
    json_path = os.path.join("gesture_logs", json_name)
    os.makedirs("gesture_logs", exist_ok=True)
    
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"💾 Zapisano gest jako JSON → {json_path}")
