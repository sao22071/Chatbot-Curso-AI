#Se inicia montando las librerias 

# Importar las librerias
import nltk
#descargamos las stopwords
nltk.download('punkt_tab')
import pandas as pd 
import numpy as np

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import random 
import json 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

#cargar los datos
path = "/content/drive/MyDrive/Chatbot/intents.json"
with open(path, 'r', encoding='utf-8') as file:
  data = json.load(file)

#Creamos el stemmer
stemmer = PorterStemmer()

#Preprocesamiento
vocab = [] #=> vocabulario
tags = []
patterns = []
labels = []

for intent in data['intents']:
  for pattern in intent['patterns']:
    tokens = word_tokenize(pattern.lower())
    #estimación -> recorta las palabras para que sean mas genericas
    stemmed = [ stemmer.stem(w) for w in tokens]
    vocab.extend(stemmed) # --> extend envia la lista que vaya recorriendo, y va llenando el array vocab
    patterns.append(stemmed)
    labels.append(intent['tag']) #en labels se guarda un tag una vez tenga las palabras e identifique esas palabras
  if intent['tag'] not in tags:
    tags.append(intent['tag']) #si tags no tiene ese tag entonces que lo agregue

#que convierta la lista vocab a un conjunto para que solo sea una palabra y no varias 
vocab = sorted(set(vocab))

X=[]
Y=[]

encoder = LabelEncoder()
encoder_labels = encoder.fit_transform(labels)

# pattern es la lista con el archivo json
# 0 en vocab es porque ya existe en el glosario(vocab)
#Tecnica para covertir categoria a una binaria
for pattern in patterns:
  bag = [1 if word in pattern else 0 for word in vocab]
  X.append(bag)


Y = encoder_labels #Lo primero que hace es identificar la etiqueta adecuada para que diga a que pertenece 

#Convertimos las variables a arreglos de numpy 

X = np.array(X) #Cantidad de datos en cada fila
Y = np.array(Y)

#Modelo 
D = len(X[0]) #Cantidad de entradas
C = len(X[0]) #Cantidad de etiquetas

model = Sequential()
#Capa de entrada- densa, capa densa con 8 neuronas, la coma (,) es
# D es el tamaño de las filas que entraron 
# para que el mismo programa identifique la cantidad de salida que tendrá
#activiation, reconocer la información y operaciones matemáticas
model.add(Dense(8, input_shape = (D,), activation = 'relu'))

#Capa densea 2
model.add(Dense(8, activation = 'relu'))


#tratar en lo posible asignar una neurona a cada categoria, y todo depende de la cantidad de categorias
model.add(Dense(C, activation='softmax')) 

#Compilamos
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics= ['accuracy']
)

#Entrenamos 
#verbose = 0, es para que no muestre nada en consola 
model.fit(X,Y, epochs = 200, verbose=0)


#Función para procesar la entrada --> lo que escriba el usuario en el chat 
def predict_class(text):
  tokens = word_tokenize(text.lower())
  stemmed = [stemmer.stem(w) for w in tokens]
  bag = np.array([1 if word in stemmed else 0 for word in vocab])
  res = model.predict(np.array([bag]), verbose = 0)[0] #seleccione las palabras que encontró
  idx = np.argmax(res)
  #decodificación de la etiqueta -para que entrege la palabra y no el numero
  tag = encoder.inverse_transform([idx])[0]
  return tag

#Funcion para dar la respuesta
def get_response(tag, context):
  for intent in data['intents']:
    if intent['tag'] == tag:
      return random.choice(intent['responses'])
  return "No entendí eso, ¿Puede repetirlo?"

















































