# 2️⃣ Skrypt trenujący model – train_model.py
# 📂 Opis:
#
# Ładuje przygotowane dane
#
# Tworzy i trenuje model LSTM do rozpoznawania gestów
#
# Zapisuje wytrenowany model do pliku .h5


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding etykiet (dla klasyfikacji wieloklasowej)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Budowa modelu LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(len(gestures), activation="softmax")  # Liczba gestów = liczba neuronów
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Zapis modelu
model.save("gesture_recognition_model.h5")

print("✅ Model wytrenowany i zapisany!")
