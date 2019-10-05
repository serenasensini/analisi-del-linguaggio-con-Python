import speech_recognition as sr

recognizer_instance = sr.Recognizer()
wav = sr.AudioFile("/path/to/file.wav")

with wav as source:
    recognizer_instance.pause_threshold = 3.0
    audio = recognizer_instance.listen(source)
    print("Messaggio registrato. Elaborazione in corso!")
try:
    text = recognizer_instance.recognize_google(audio, language="it-IT")
    print("Il testo sembrerebbe essere: \n", text)
except Exception as e:
    print(e)