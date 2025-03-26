# 3️⃣ Skrypt do klasyfikacji gestów – predict_gesture.py
# 📂 Opis:
#
# Wczytuje wytrenowany model
#
# Otrzymuje nowe sekwencje ruchów dłoni
#
# Przewiduje, który gest został wykonany

import numpy as np
from tensorflow.keras.models import load_model

# Wczytanie modelu
model = load_model("gesture_recognition_model.h5")


def predict_gesture(sequence):
    """
    Przewiduje gest na podstawie zarejestrowanej sekwencji ruchów.

    :param sequence: Sekwencja [X, Y, Z] o stałej liczbie klatek
    :return: Nazwa rozpoznanego gestu
    """
    sequence = np.expand_dims(sequence, axis=0)  # Dopasowanie wymiarów
    prediction = model.predict(sequence)
    label_index = np.argmax(prediction)

    return list(gestures.keys())[label_index]


# Przykładowe testowanie
test_sequence = X_test[0]  # Weź próbkę testową
recognized_gesture = predict_gesture(test_sequence)
print(f"🤖 Rozpoznano gest: {recognized_gesture}")
