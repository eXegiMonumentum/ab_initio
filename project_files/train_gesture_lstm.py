"""
1️⃣ Nagrywanie danych (gesture_record.py)
2️⃣ Przygotowanie danych (prepare_gesture_data.py)
3️⃣ Trening modelu (train_gesture_lstm.py)
4️⃣ Ewaluacja modelu (evaluate_model.py)
5️⃣ Rozpoznawanie gestów na żywo z kamery (predict_live.py)
"""

""""
!!!
Ten skrypt zakłada, że wcześniej uruchomiłeś prepare_gesture_data.py, który zapisał:

X_gestures.npy – sekwencje (num_samples, 100, 63),
y_gestures.npy – etykiety (num_samples,)
!!!

Trenuje model LSTM do klasyfikacji gestów dłoni na podstawie danych 3D (x, y, z) z MediaPipe.
Dane wejściowe powinny mieć format: (num_samples, sequence_length, 63),
gdzie 63 to 21 punktów dłoni * 3 współrzędne.

Wymagania:
- NumPy
- PyTorch
- Scikit-learn

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# === Wczytaj dane z plików .npy zapisanych wcześniej ===
X = np.load("X_gestures.npy")  # shape: (num_samples, sequence_length, 63)
y = np.load("y_gestures.npy")  # shape: (num_samples,)

# Konwersja do tensora PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# === Podział na dane treningowe i testowe ===
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# === DataLoadery do batchowania danych ===
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)


# === Definicja modelu LSTM ===
class GestureLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # tylko ostatnia klatka czasowa
        return self.fc(out)


# === Inicjalizacja modelu, funkcji straty i optymalizatora ===
model = GestureLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Trening modelu ===
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoka {epoch + 1} - Strata: {loss.item():.4f}")

# === Zapisz wytrenowany model do pliku ===
model_path = "gesture_model.pt"
torch.save(model.state_dict(), model_path)
print(f"💾 Model zapisany jako {model_path}")

# === Ewaluacja skuteczności modelu ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"🎯 Skuteczność modelu: {accuracy:.2f}%")
