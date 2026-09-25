import speech_recognition as sr


LANGUAGES = {
    "1": ("English", "en-IN"),
    "2": ("Hindi", "hi-IN"),
    "3": ("Kannada", "kn-IN")
}


def speech_to_text(language_code):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(
            audio,
            language=language_code
        )

        return text

    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
        return ""

    except sr.RequestError as e:
        print("❌ Speech recognition service error:", e)
        return ""


if __name__ == "__main__":

    print("\n==============================")
    print("      SPEECH TO TEXT")
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

        text = speech_to_text(language_code)

        if text:
            print("\nRecognized Text:")
            print(text)