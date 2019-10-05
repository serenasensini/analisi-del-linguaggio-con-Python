import speech_recognition as sr

recognizer_instance = sr.Recognizer()

with sr.Microphone() as source:
    recognizer_instance.adjust_for_ambient_noise(source)
    print("Sono pronto, puoi parlare")
    audio = recognizer_instance.listen(source)
    print("Messaggio registrato. Elaborazione in corso!")
try:
    text = recognizer_instance.recognize_google(audio, language="it-IT")
    print("Il testo sembrerebbe essere: \n", text)
except Exception as e:
    print(e)