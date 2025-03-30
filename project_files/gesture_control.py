import pyautogui

def handle_gesture(gesture_name):
    """
    Przyjmuje nazwę rozpoznanego gestu i wykonuje odpowiadający skrót klawiszowy w systemie.
    """
    gesture_to_action = {
        "shake": lambda: pyautogui.hotkey('ctrl', 'a'),           # Zaznacz wszystko
        "stop": lambda: pyautogui.press('space'),                 # Zatrzymaj / Odznacz
        "wave": lambda: pyautogui.hotkey('ctrl', 's'),            # Zapisz
        "wave_both": lambda: pyautogui.hotkey('ctrl', 'x'),      # Zapisz i wyjdź
        "freeze": lambda: pyautogui.press('f'),                   # Zamroź segment
        "wiggle_v": lambda: pyautogui.press('w'),                   # Optymalizacja (Wiggle)
        "shake_sidechains": lambda: pyautogui.press('s'),         # Potrząśnij łańcuchami bocznymi
        "rebuild": lambda: pyautogui.press('o'),                  # Przebuduj strukturę
        "zoom_in": lambda: pyautogui.press('pageup'),             # Zoom +
        "zoom_out": lambda: pyautogui.press('pagedown')           # Zoom -
    }

    if gesture_name in gesture_to_action:
        print(f"🎯 Wykryto gest: {gesture_name} → wykonuję akcję.")
        gesture_to_action[gesture_name]()
    else:
        print(f"❌ Nierozpoznany gest: {gesture_name}")


## przykład użycia:

# from gesture_control import handle_gesture
#
# # np. wynik modelu
# predicted_gesture = "shake"
#
# handle_gesture(predicted_gesture)
