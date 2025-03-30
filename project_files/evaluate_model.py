"""
1️⃣ Nagrywanie danych (gesture_record.py)
2️⃣ Przygotowanie danych (prepare_gesture_data.py)
3️⃣ Trening modelu (train_gesture_lstm.py)
4️⃣ Ewaluacja modelu (evaluate_model.py)
5️⃣ 🔜 (opcjonalnie) Rozpoznawanie gestów na żywo z kamery (predict_live.py)

"""

"""
Ten skrypt służy do oceny skuteczności wytrenowanego modelu LSTM
do rozpoznawania gestów dłoni na podstawie danych 3D (MediaPipe).

Wczytuje dane testowe (X_gestures.npy, y_gestures.npy) oraz model (.pt),
a następnie oblicza dokładność, macierz pomyłek i raport klasyfikacji.


"""
#  Co musisz mieć wcześniej:
# X_gestures.npy, y_gestures.npy → z prepare_gesture_data.py
#
# gesture_model.pt → wytrenowany i zapisany model z train_gesture_lstm.py
#
# zdefiniowaną klasę GestureLSTM w pliku train_gesture_lstm.py

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# === Wczytaj dane wejściowe i etykiety ===
# Pliki X_gestures.npy i y_gestures.npy powinny zostać wygenerowane przez prepare_gesture_data.py
X = np.load("X_gestures.npy")  # shape: (num_samples, sequence_length, 63)
y = np.load("y_gestures.npy")  # shape: (num_samples,)

# === Podział danych na zestaw treningowy i testowy ===
# Używamy tylko danych testowych do ewaluacji modelu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Konwersja danych testowych do tensorów PyTorch ===
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# === Wczytanie architektury modelu LSTM ===
# Zakładamy, że GestureLSTM jest zdefiniowany w pliku train_gesture_lstm.py
from train_gesture_lstm import GestureLSTM

# === Utworzenie modelu i wczytanie jego wag z pliku ===
model = GestureLSTM()
model.load_state_dict(torch.load("gesture_model.pt"))  # Upewnij się, że masz ten plik
model.eval()  # Przełączenie modelu w tryb ewaluacyjny

# === Predykcja etykiet na danych testowych ===
with torch.no_grad():  # Bez obliczania gradientów
    outputs = model(X_test_tensor)  # Wyjście modelu: predykcje (logity)
    predicted = torch.argmax(outputs, dim=1).numpy()  # Wybór klasy z najwyższym wynikiem

# === Ocena dokładności modelu ===
accuracy = accuracy_score(y_test, predicted)
print(f"🎯 Dokładność: {accuracy * 100:.2f}%")

# === Szczegółowy raport klasyfikacji ===
print("\n🧾 Raport klasyfikacji:")
print(classification_report(y_test, predicted))  # Precision, recall, F1-score

# === Macierz pomyłek – ile razy pomylono daną klasę ===
print("📉 Macierz pomyłek:")
cm = confusion_matrix(y_test, predicted)

# === Wizualizacja macierzy pomyłek jako heatmapa ===
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywista klasa")
plt.title("📊 Macierz pomyłek gestów")
plt.show()
