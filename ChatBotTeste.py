import json as js
import numpy as np
import nltk
from nltk.stem.rslp import RSLPStemmer
import tensorflow as tf
import tflearn as tfl
import random
import os
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

# Carregando Json
with open("intents.json", encoding='utf-8') as file:
  data = js.load(file)
#Preparando os dados  
nltk.download('rslp')
nltk.download('punkt')
palavras = []
intencoes = []
sentencas = []
saidas = []

def cria_audio(text):
  tts = gTTS(text,lang = 'pt')
  #Salva o arquivo de audio
  caminho = 'audios/001.mp3'
  tts.save(caminho)
 
  #play ao audio
  playsound(caminho) 
  os.remove(caminho)
  
def ouvir_microfone():
    frase=""
	#Habilita o microfone para ouvir o usuario
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
		#Chama a funcao de reducao de ruido disponivel na speech_recognition
        microfone.adjust_for_ambient_noise(source)
				
		#Armazena a informacao de audio na variavel
        audio = microfone.listen(source)
    try:
        
		#Passa o audio para o reconhecedor de padroes do speech_recognition
        frase = microfone.recognize_google(audio,language='pt-BR')
        
		#Caso nao tenha reconhecido o padrao de fala, exibe esta mensagem
        
    except sr.UnknownValueError:
            frase = cria_audio("Não entendi")
        
    return frase


for intent in data["intents"]:
    
  tag = intent['tag']

  if tag not in intencoes:
     intencoes.append(tag)

  for pattern in intent["patterns"]:
    wrds = nltk.word_tokenize(pattern, language='portuguese')
    palavras.extend(wrds)
    sentencas.append(wrds)
    saidas.append(tag)
#Stemming
stemer = RSLPStemmer()

stemmed_words = [stemer.stem(w.lower()) for w in palavras]
stemmed_words = sorted(list(set(stemmed_words)))
#Bag of Words
training = []
output = []
outputEmpty = [0 for _ in range(len(intencoes))]

for x, frase in enumerate(sentencas):
  bag = []
  wds = [stemer.stem(k.lower()) for k in frase]
  for w in stemmed_words:
    if w in wds:
      bag.append(1)
    else:
      bag.append(0)

  outputRow = outputEmpty[:]
  outputRow[intencoes.index(saidas[x])] = 1

  training.append(bag)  
  output.append(outputRow)

#Rede Neural
training = np.array(training)
output = np.array(output)

net = tfl.input_data(shape=[None, len(training[0])])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(output[0]), activation="softmax")
net = tfl.regression(net)
model = tfl.DNN(net)

#Treinamento
model.fit(training, output, n_epoch=30, batch_size=8, show_metric=True)
model.save("Modelo_Bot")

#Bot
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

def chat():
    cria_audio('Oi, sou a Julia!! Vamos conversar?')
    Online = True
    while Online:
        inp = ouvir_microfone()
        bag_usuario = bag_of_words(inp, stemmed_words)
        results = model.predict([bag_usuario])
        results_index = np.argmax(results)
        tag = intencoes[results_index]
        maximo=results.max()
        if maximo>0.15:
         for tg in data["intents"]:
             if tg['tag'] == tag:
                 responses = tg['responses']

         cria_audio(random.choice(responses))

         if tag == "ate-mais":
             Online = False
        else:
          cria_audio('Não entendi, pode repetir?')
chat()

  