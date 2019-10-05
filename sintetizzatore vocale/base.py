from gtts import gTTS
import subprocess

testo = """Ciao, sono il tuo assistente digitale!"""

tts = gTTS(text= testo, lang='it')

tts.save("result.mp3")

print("That’s all, folks!")

subprocess.run(["audacious", "result.mp3"])

