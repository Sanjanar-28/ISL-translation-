from gtts import gTTS
import pygame
import os
import time


LANGUAGES = {
    "1": ("English", "en"),
    "2": ("Hindi", "hi"),
    "3": ("Kannada", "kn")
}


def text_to_speech(text, language_code):

    filename = "speech_output.mp3"

    tts = gTTS(
        text=text,
        lang=language_code,
        slow=False
    )

    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()

    os.remove(filename)


if __name__ == "__main__":

    print("\n==============================")
    print("      TEXT TO SPEECH")
    print("==============================")
    print("1. English")
    print("2. Hindi")
    print("3. Kannada")

    choice = input("\nSelect language: ")

    if choice not in LANGUAGES:
        print("❌ Invalid choice.")
    else:

        language_name, language_code = LANGUAGES[choice]

        print(f"\nSelected language: {language_name}")

        text = input("Enter text: ")

        if text.strip():
            print("\n🔊 Speaking...")
            text_to_speech(text, language_code)