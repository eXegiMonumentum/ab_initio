






# 🖐 Real-Time Hand Gesture Recognition System
> Kamera OAK-D + MediaPipe + PyTorch LSTM  
> Autorzy: *[Tenerowicz Filip, Marczak Bogdan]*  
> Gesty: [Skróty klawiszowe Foldit.docx](Skr%F3ty%20klawiszowe%20Foldit.docx)  
> *[opisane przez Mikus Wiktorię oraz Peretiako Laurę]*  
> Status: 🚀 Stabilny / w ciągłym rozwoju

### 🚀 Status: Stabilny / w ciągłym rozwoju

## 🧠 Opis projektu

System umożliwia rozpoznawanie gestów dłoni w czasie rzeczywistym przy użyciu heurystyk geometrycznych. Służy do interakcji z aplikacjami, sterowania kursorem myszy oraz symulowania kliknięć – bez użycia uczenia maszynowego.

Obsługiwane są dwa tryby:

- **Tryb gestów** – rozpoznaje statyczne układy dłoni na podstawie ich pozycji i kształtu.
- **Tryb myszy** – umożliwia sterowanie kursorem i klikaniem na podstawie układu dłoni.

---

## 📌 Kluczowe komponenty

### 📷 Kamera i detekcja
- Strumień wideo z kamery OAK-D (1080p @ 30 FPS).
- Wykrywanie landmarków dłoni za pomocą MediaPipe.
- Automatyczne przypisanie dłoni jako lewej lub prawej.

### 🔍 Heurystyki gestów
- Reguły geometryczne rozpoznające ułożenie palców i dłoni (pięść, STOP, wskazujący, OK).
- Bufor ostatnich klatek + stabilizacja rozpoznania.

---

## 🖱 Tryb Myszki (Mouse Mode)

### 🔄 Aktywacja / dezaktywacja
- Lewa pięść wystawiona w lewo przez **4 sekundy** – przełącza tryb myszy.

### 🎯 Sterowanie
- **Lewa ręka (wskazujący palec)** – steruje kursorem.
- **Prawa ręka**:
  - Gest OK: **kliknięcie**.
  - Pięść: **mouseDown** (przytrzymanie).
  - Rozluźnienie pięści: **mouseUp**.
- Lewa ręka w pozycji "STOP": blokuje kliknięcia.

### 🖥 Overlay
- Komunikaty na ekranie: `"MOUSE MODE: ENABLED"` / `"GESTURE MODE: ENABLED"`.

---

## ✋ Tryb Gestów (Gesture Mode)

| Gest        | Warunek                                 |
|-------------|------------------------------------------|
| `ZOOM_IN`   | Obie ręce w pięści, blisko siebie        |
| `ZOOM_OUT`  | Obie ręce w pięści, szeroko rozstawione  |
| `STOP`      | Lewa ręka płasko wyprostowana z lewej strony |

- Stabilizacja: gest musi być utrzymany przez kilka klatek.
- Odliczanie (3 sekundy) przed aktywacją.
- Licznik wykonanych gestów.

---

Projekt stworzony z pasji.  
Zaprojektowany modularnie, gotowy na rozwój i deployment.



Masz pytania lub chcesz współtworzyć?  
E-mail: [167128@stud.prz.edu.pl](mailto:167128@stud.prz.edu.pl)

Lub zgłoś się przez **GitHub!** ⭐


