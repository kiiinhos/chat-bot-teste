import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy
import tensorflow as tf
import random
import json
import tflearn
import pickle


with open("intents.json") as file:
    data = json.load(file)
    
try:
    with open('data.pickle','rb') as f:
          palavras,linhas,treinador,output = pickle.load(f)
except:

    palavras=[]
    linhas = []
    docs_x = []
    docs_y = []


for intent in data ["intents"]:
    for pattern in intent["patterns"]:
        palavra = nltk.word_tokenize(pattern)
        palavras.extend(palavra)
        docs_x.append(palavra)
        docs_y.append(intent["tag"])

        if intent["tag"] not in linhas:
            linhas.append(intent["tag"])

palavras = [stemmer.stem(w.lower())for w in palavras if w not in "?"]
palavras = sorted(list(set(palavras)))


linhas = sorted(linhas)

treinador = []
output = []

out_vazio = [0 for _ in range(len(linhas))]

for x, doc in enumerate(docs_x):
    bolsa =[]

    palavra = [stemmer.stem(w) for w in doc]

    for w in palavras:
        if w in palavra:
            bolsa.append(1)
        else:
            bolsa.append(0)

    output_row = out_vazio[:]
    output_row[linhas.index(docs_y[x])] = 1

    treinador.append(bolsa)
    output.append(output_row)

treinador = numpy.array(treinador)
output = numpy.array(output)

with open('data.pickle','wb') as f:
        pickle.dump((palavras,linhas,treinador,output),f)


tf.reset_default_graph()

net = tflearn.input_data(shape=[None,len(treinador[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation='softmax')
net= tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load('model.tflearn')
except:

    model.fit(treinador,output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bolsa_de_palavras(s,palavras):
    bolsa = [0 for _ in range(len(palavras))]

    s_palavras = nltk.word_tokenize(s)
    s_palavras = [stemmer.stem(palavra.lower())for palavra in s_palavras]

    for se in s_palavras:
        for i, w in enumerate(palavras):
            if w == se:
                bolsa[i] = 1

    return numpy.array(bolsa)


def chat():

    print('Comece a fala com o Bot! (Escreva sair)')
    while True:

        inp = input('Voce:')
        if inp.lower() == 'sair':
            break

        resultados = model.predict([bolsa_de_palavras(inp,palavras)])
        resultados_index = numpy.argmax(resultados)
    etiqueta = linhas[resultados_index]
       
    for etiquet in data['intents']:
           if etiquet['tag'] == etiqueta:
               responses = etiquet['responses']

    print(random.choice(responses))

chat()