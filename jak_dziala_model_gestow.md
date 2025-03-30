
# 🧠 Jak model rozpoznaje różne gesty?

---

## 🔍 Model wie, który gest jest który dzięki etykietom (`y`)

W `prepare_gesture_data.py` kluczowa sekcja:

```python
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
```

Dla każdego gestu:
- `shake` dostaje etykietę `0`
- `wave` dostaje etykietę `1`

Potem, jak skrypt przetwarza CSV-y z katalogu `gestures_data/shake/`, to do każdego przykładu przypisuje label `0`.

**Dane wynikowe:**
- `X` – przykłady: np. 60 sekwencji o kształcie `(100, 63)`
- `y` – etykiety: np. `[0, 0, 0, ..., 1, 1, 1]`

---

## 🧠 Trenowanie modelu uczy go rozróżniać te etykiety

W `train_gesture_lstm.py`, model trenuje się tak:

```python
outputs = model(batch_X)  # prognozy dla batcha
loss = criterion(outputs, batch_y)  # porównanie z prawdziwą etykietą
```

Model patrzy na sekwencję 3D punktów dłoni i uczy się:
> „Aha, jeśli palce drgają w ten sposób → to shake (0), a jeśli falują → to wave (1)”.

---

## 🔎 W czasie predykcji na żywo (`predict_live.py`)

- Model analizuje sekwencję `100` klatek (czyli `100x63` wejść).
- Wypluwa wynik np. `[0.01, 0.98]` → czyli prawdopodobieństwo dla klas `shake`, `wave`.
- Z `torch.argmax()` wybiera najbardziej prawdopodobną klasę.

---

## ✅ Podsumowanie

➡️ Wszystkie dane są ładowane naraz, **ale z przypisaną etykietą (klasą)**, więc model wie, co jest czym.  
➡️ Trening uczy model różnic między ruchami.  
➡️ Predykcja sprawdza, który ruch jest najbardziej podobny do zapamiętanych gestów.
