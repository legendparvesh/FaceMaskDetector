
import speech_recognition as sr
from google_trans_new import google_translator
r=sr. Recognizer()
translator = google_translator()
text=translator.translate('turn left',lang_tgt='ta')
#with sr.Microphone() as source:
    #print("Speak now!" )
    #audio = r. listen(source)
    #try:
        #speech_text= r. recognize_google(audio)
        #print (speech_text)
    #except sr.UnknownVa1ueError:
        #print("Cou1d not understand")
    #except sr.RequestError:
        #print("Cou1d not request result from google")
    #translated_text = translator. translate('turn left',lang_tgt='ta')