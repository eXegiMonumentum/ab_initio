# ✋ Słownik gestów – sterowanie Foldit

Ten dokument zawiera opis gestów ręki używanych do sterowania grą **Foldit** za pomocą systemu rozpoznawania gestów (MediaPipe + LSTM).

Każdy gest został zaprojektowany tak, by był łatwy do wykonania, wyraźny dla kamery i jednoznaczny dla modelu predykcyjnego.

---

## 📘 Lista gestów i przypisanych skrótów Foldit:

### 👋 `shake` – **Ctrl + A**  
Zaznacz wszystko  
**Gest:** Potrząśnięcie jedną otwartą dłonią w przód–tył (jak "nie, nie, nie")

---

### ✋ `stop` – **Spacja**  
Zatrzymaj akcję lub odznacz segmenty  
**Gest:** Klaśnięcie obu dłoni na wysokości twarzy

---

### 🖐️ `wave` – **Ctrl + S**  
Zapisz aktualne rozwiązanie  
**Gest:** Lewa ręka: uniesienie powyżej twarzy → zaciśnięcie pięści → opuszczenie

---

### 🙌 `wave_both` – **Ctrl + X**  
Zapisz i wyjdź  
**Gest:** Obie ręce: uniesienie powyżej twarzy → zaciśnięcie obu pięści → opuszczenie

---

### 🧊 `freeze` – **F**  
Zamrożenie segmentu  
**Gest:** Wyciągnięcie palca wskazującego (reszta palców zgięta) w kierunku kamery

---

### ✌️ `wiggle_V` – **W**  
Optymalizacja struktury  
**Gest:** Rotacja barku w górę → pokazanie litery „V” palcami (wskazujący + środkowy)

---

### 🤲 `shake_sidechains` – **S**  
Potrząśnięcie łańcuchami bocznymi  
**Gest:** Obie dłonie z rozstawionymi palcami, poruszane w górę i w dół (jak strzepywanie wody)

---

### 🌀 `rebuild` – **O**  
Przebuduj fragment  
**Gest:** Rysowanie okręgu lewą otwartą dłonią w powietrzu (zgodnie z ruchem wskazówek zegara)

---

### 🔍 `zoom_in` – **Page Up**  
Przybliżenie  
**Gest:** Dwie pięści blisko siebie → powolne rozsunięcie ich na boki (jak rozciąganie gumki)

---

### 🔎 `zoom_out` – **Page Down**  
Oddalenie  
**Gest:** Dwie pięści lekko rozstawione → powolne zbliżenie ich do siebie (jakby składać gumkę)

---

## 📂 Uwagi:

- Wszystkie gesty powinny być wykonywane **na wysokości klatki piersiowej lub twarzy**.
- Gesty zostały zaprojektowane tak, by **unikać przypadkowych aktywacji** i zapewniać maksymalną rozpoznawalność.
- Długość trwania gestu: zalecane ok. **1–2 sekundy**.
- Ruchy powinny być **płynne, ale zdecydowane**.

---

## 📌 Wymagania systemowe:

- Kamera OAK-D lub inne urządzenie kompatybilne z DepthAI
- Python 3.8+
- MediaPipe, PyTorch, OpenCV, DepthAI (szczegóły w `requirements.txt`)

---