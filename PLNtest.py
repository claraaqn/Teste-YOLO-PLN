import speech_recognition as sr

# Pra funcionar precisa intalar a biblioteca SpeechRecognition com o comando: pip install SpeechRecognition
# E também a biblioteca PyAudio com o comando: python -m pip install pyaudio

recon = sr.Recognizer()
# Caso preciso escolher um microfone diferente rode esse código para saber todos os microfones conectados
#print(sr.Microphone.list_microphone_names())
# é um array no final só precisa colocar o índice do microfone que deseja usar como sr.Micorphone(device_index=0,1,2,3,...)
with sr.Microphone() as source:
    print("Diga alguma coisa")
    audio = recon.listen(source)

print(recon.recognize_google(audio, language='pt-BR'))