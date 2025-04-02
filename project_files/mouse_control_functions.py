import numpy as np

def is_fist(landmarks):
    folded = lambda tip, pip: landmarks[tip].y > landmarks[pip].y
    return all(folded(t, p) for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)])
def is_stop(landmarks):
    """
    Funkcja sprawdza, czy lewa ręka jest w pozycji "STOP" (wszystkie palce wyprostowane).
    """
    # Sprawdzamy, czy wszystkie palce są wyprostowane
    # Dłoń w pozycji "STOP" ma wszystkie palce wyprostowane
    extended_fingers = [
        landmarks[4].y < landmarks[3].y,  # Kciuk
        landmarks[8].y < landmarks[6].y,  # Palec wskazujący
        landmarks[12].y < landmarks[10].y,  # Palec środkowy
        landmarks[16].y < landmarks[14].y,  # Palec serdeczny
        landmarks[20].y < landmarks[18].y   # Palec mały
    ]
    # Jeśli wszystkie palce są wyprostowane, to zwróć True
    return all(extended_fingers)
def is_pointing(landmarks):
    extended_index = landmarks[8].y < landmarks[6].y
    folded_others = all(landmarks[tip].y > landmarks[pip].y for tip, pip in [(12, 10), (16, 14), (20, 18)])
    return extended_index and folded_others
def is_extended(landmarks):
    extended_fingers = [
        landmarks[4].y < landmarks[3].y,  # Kciuk
        landmarks[8].y < landmarks[6].y,  # Wskazujący
        landmarks[12].y < landmarks[10].y,  # Środkowy
        landmarks[16].y < landmarks[14].y,  # Serdeczny
        landmarks[20].y < landmarks[18].y  # Mały
    ]
    return all(extended_fingers)  # Jeśli wszystkie palce są wyprostowane
def is_okay_gesture(landmarks):
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    touching = distance(thumb_tip, index_tip) < 0.05
    middle_extended = middle_tip.y < middle_pip.y
    return touching and middle_extended
