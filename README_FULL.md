# 🖐 Real-Time Hand Gesture Recognition System  
> Kamera OAK-D + MediaPipe + PyTorch LSTM  
> Autorzy: *[Pojawią się :D]*  
> Status: 🚀 Stabilny / w ciągłym rozwoju

## 🧠 O projekcie

System rozpoznawania gestów dłoni w czasie rzeczywistym, oparty na:
- detekcji punktów dłoni (MediaPipe),
- nagrywaniu i przygotowywaniu danych gestów (OAK-D),
- trenowaniu modelu LSTM (PyTorch),
- rozpoznawaniu gestów na żywo z kamery.

System może być wykorzystywany do sterowania aplikacjami, interfejsami użytkownika, w środowiskach VR/AR, lub jako rozwiązanie dla systemów embedded.

## 🎯 Główne funkcje

✅ Nagrywanie własnych gestów z kamery OAK-D  
✅ Detekcja i podpisywanie dłoni ("Left/Right")  
✅ Eksport danych do CSV + nagrania RGB/Depth AVI  
✅ Automatyczna interpolacja, normalizacja i konwersja do `.npy`  
✅ Trenowanie modelu LSTM do klasyfikacji gestów  
✅ Ewaluacja modelu  
✅ Rozpoznawanie gestów na żywo z buforem 100 klatek

## 📁 Struktura projektu

```bash
📦 gesture-recognition
├── gestures_data/           # 📂 Nagrania i dane gestów
│   ├── shake/
│   │   ├── shake_01_positions.csv
│   │   ├── shake_rgb.avi
│   │   └── shake_depth.avi
│   └── wave/
│       └── ...
├── gesture_record.py        # 🎥 Nagrywanie gestów z kamery
├── prepare_gesture_data.py  # 🧹 Przetwarzanie CSV do formatu sieci
├── train_gesture_lstm.py    # 🧠 Trenowanie modelu LSTM
├── evaluate_model.py        # 📊 Ocena skuteczności
├── predict_live.py          # 🚀 Rozpoznawanie gestów na żywo
├── gesture_record_help.py   # 🧰 Funkcje pomocnicze
├── X_gestures.npy           # 💾 Dane wejściowe (sekwencje)
├── y_gestures.npy           # 💾 Etykiety (klasy gestów)
├── gesture_model.pt         # 🧠 Wytrenowany model PyTorch
└── README.md                # 📄 Ten plik
```

## 🔁 Pipeline (przepływ pracy)

1. **[1] `gesture_record.py`**  
   👉 Nagrywaj własne gesty – kamera OAK-D, MediaPipe, CSV, RGB + Depth.

2. **[2] `prepare_gesture_data.py`**  
   👉 Interpoluj dane, normalizuj, konwertuj do `.npy` (X, y).

3. **[3] `train_gesture_lstm.py`**  
   👉 Trenuj model LSTM na przygotowanych danych.

4. **[4] `evaluate_model.py`**  
   👉 Sprawdź skuteczność modelu.

5. **[5] `predict_live.py`**  
   👉 Włącz rozpoznawanie gestów w czasie rzeczywistym!

## 🧩 Obsługiwane gesty

| Gest | Klasa |
|------|--------|
| `shake` | 0 |
| `wave` | 1 |
| `stop` | 2 |
| `wave_both` | 3 |
| `freeze` | 4 |
| `wiggle_V` | 5 |
| `shake_sidechains` | 6 |
| `rebuild` | 7 |
| `zoom_in` | 8 |
| `zoom_out` | 9 |
 0 |
| `wave`  | 1 |
| `...`   | Możesz dodawać kolejne! |

## 🛠️ Wymagania

- Python 3.8+
- PyTorch
- DepthAI
- MediaPipe
- NumPy, OpenCV, Matplotlib, Scikit-learn

Instalacja (pip):
```bash
pip install torch opencv-python mediapipe numpy matplotlib scikit-learn
```

## 🛣️ Plany rozwoju

- [x] Obsługa wielu gestów
- [ ] Rozpoznawanie obu dłoni jednocześnie
- [ ] Sterowanie aplikacjami (Foldit, VR, GUI)
- [ ] Eksport do embedded/mobile (Raspberry Pi, Android)
- [ ] Integracja z Google VR i środowiskami 3D

## 🧠 Model

LSTM przyjmuje dane wejściowe w formacie `(samples, 100 klatek, 63 współrzędnych)`  
Każda próbka zawiera 21 punktów dłoni × 3 współrzędne (x, y, z) na każdą klatkę.

## 🖼️ Przykładowa animacja (prepare_gesture_data.py)

⬆️ Interpolowana trajektoria dłoni w 3D (**dodam grafikę**)

## 🧑‍💻 Autorzy
    --> pojawią się :D

Projekt stworzony z pasji. 
Zaprojektowany modularnie, gotowy na rozwój i deployment.

Masz pytania lub chcesz współtworzyć?  
e-mail: 167128@stud.prz.edu.pl

lub zgłoś się przez GitHub! ⭐
