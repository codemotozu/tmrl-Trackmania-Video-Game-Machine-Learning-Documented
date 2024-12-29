# standard library imports  # Import von Standardbibliotheken
import platform  # Module to access system information.  # Modul für Zugriff auf Systeminformationen

if platform.system() in ("Windows", "Linux"):  # Check if the system is Windows or Linux.  # Überprüfen, ob das System Windows oder Linux ist

    import time  # Module for time-related functions.  # Modul für zeitbezogene Funktionen

    def control_gamepad(gamepad, control):  # Function to control the gamepad.  # Funktion zur Steuerung des Gamepads
        assert all(-1.0 <= c <= 1.0 for c in control), "This function accepts only controls between -1.0 and 1.0"  # Ensure all controls are within valid range.  # Stellt sicher, dass alle Steuerungen im gültigen Bereich liegen
        if control[0] > 0:  # gas  # Wenn Steuerung [0] positiv ist: Gas geben
            gamepad.right_trigger_float(value_float=control[0])  # Apply right trigger value for acceleration.  # Rechten Trigger für Beschleunigung betätigen
        else:
            gamepad.right_trigger_float(value_float=0.0)  # Release right trigger if no acceleration.  # Rechten Trigger loslassen, wenn keine Beschleunigung
        if control[1] > 0:  # break  # Wenn Steuerung [1] positiv ist: Bremsen
            gamepad.left_trigger_float(value_float=control[1])  # Apply left trigger value for braking.  # Linken Trigger für Bremsen betätigen
        else:
            gamepad.left_trigger_float(value_float=0.0)  # Release left trigger if no braking.  # Linken Trigger loslassen, wenn kein Bremsen
        gamepad.left_joystick_float(control[2], 0.0)  # turn  # Joystick links: Drehen steuern
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren

    def gamepad_reset(gamepad):  # Function to reset the gamepad.  # Funktion zum Zurücksetzen des Gamepads
        gamepad.reset()  # Reset the gamepad state.  # Gamepad-Status zurücksetzen
        gamepad.press_button(button=0x2000)  # press B button  # B-Taste drücken
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.1)  # Wait for 0.1 seconds.  # 0,1 Sekunden warten
        gamepad.release_button(button=0x2000)  # release B button  # B-Taste loslassen
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren

    def gamepad_save_replay_tm20(gamepad):  # Function to save replay in Trackmania 2020.  # Funktion zum Speichern eines Replays in Trackmania 2020
        time.sleep(5.0)  # Wait for 5 seconds.  # 5 Sekunden warten
        gamepad.reset()  # Reset the gamepad state.  # Gamepad-Status zurücksetzen
        gamepad.press_button(0x0002)  # dpad down  # Steuerkreuz nach unten drücken
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.1)  # Wait for 0.1 seconds.  # 0,1 Sekunden warten
        gamepad.release_button(0x0002)  # dpad down  # Steuerkreuz nach unten loslassen
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.2)  # Wait for 0.2 seconds.  # 0,2 Sekunden warten
        gamepad.press_button(0x1000)  # A  # A-Taste drücken
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.1)  # Wait for 0.1 seconds.  # 0,1 Sekunden warten
        gamepad.release_button(0x1000)  # A  # A-Taste loslassen
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.2)  # Wait for 0.2 seconds.  # 0,2 Sekunden warten
        gamepad.press_button(0x0001)  # dpad up  # Steuerkreuz nach oben drücken
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.1)  # Wait for 0.1 seconds.  # 0,1 Sekunden warten
        gamepad.release_button(0x0001)  # dpad up  # Steuerkreuz nach oben loslassen
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.2)  # Wait for 0.2 seconds.  # 0,2 Sekunden warten
        gamepad.press_button(0x1000)  # A  # A-Taste drücken
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.1)  # Wait for 0.1 seconds.  # 0,1 Sekunden warten
        gamepad.release_button(0x1000)  # A  # A-Taste loslassen
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren

    def gamepad_close_finish_pop_up_tm20(gamepad):  # Function to close finish pop-up in Trackmania 2020.  # Funktion zum Schließen des Abschluss-Pop-ups in Trackmania 2020
        gamepad.reset()  # Reset the gamepad state.  # Gamepad-Status zurücksetzen
        gamepad.press_button(0x1000)  # A  # A-Taste drücken
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren
        time.sleep(0.1)  # Wait for 0.1 seconds.  # 0,1 Sekunden warten
        gamepad.release_button(0x1000)  # A  # A-Taste loslassen
        gamepad.update()  # Update gamepad state.  # Gamepad-Status aktualisieren

else:  # If system is not Windows or Linux.  # Falls das System nicht Windows oder Linux ist:

    def control_gamepad(gamepad, control):  # Empty function for unsupported systems.  # Leere Funktion für nicht unterstützte Systeme
        pass  # Do nothing.  # Keine Aktion

    def gamepad_reset(gamepad):  # Empty function for unsupported systems.  # Leere Funktion für nicht unterstützte Systeme
        pass  # Do nothing.  # Keine Aktion

    def gamepad_save_replay_tm20(gamepad):  # Empty function for unsupported systems.  # Leere Funktion für nicht unterstützte Systeme
        pass  # Do nothing.  # Keine Aktion

    def gamepad_close_finish_pop_up_tm20(gamepad):  # Empty function for unsupported systems.  # Leere Funktion für nicht unterstützte Systeme
        pass  # Do nothing.  # Keine Aktion
