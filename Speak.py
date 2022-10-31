from win32com.client import Dispatch

def speak(str):
    speak = Dispatch('SAPI.SpVoice')
    speak.Speak(str)