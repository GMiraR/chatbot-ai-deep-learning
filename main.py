#Natural Language Toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
#Stemming to get the root of the words
stemmer = LancasterStemmer()


#To work with numpy arrays
import numpy as np


import tflearn
import tensorflow as tf
import random

# to save the preprocessing data
import pickle

#To read JSON files
import json


#Reading the JSON file
with open('intents.json') as file:
	data = json.load(file)

#### PREPROCESSING DATA #######


# Print insidade of dataset and get the dicts for python
# print(data['dataset'])
try:
	# rb = read bytes
	with open('data.pickle', 'rb') as f:
		words, labels, training, output = pickle.load(f)

except:
	words = []
	labels = []

	#To match the tags
	docs_x = []
	docs_y = []


	for dt in data['dataset']:
		for pattern in dt['patterns']:
			#Tokenize the patterns
			wrds = nltk.word_tokenize(pattern)
			#Extend to the words list
			words.extend(wrds)
			# For later comparisons between the words and the tags		
			docs_x.append(wrds)
			docs_y.append(dt['tag'])

		if dt['tag'] not in labels:
			labels.append(dt['tag'])

	# Stem the words, convert to lower case and 
	words = [stemmer.stem(w.lower()) for w in words]

	# Sort and remove duplicates
	words = sorted(list(set(words)))

	#Sort labels
	labels = sorted(labels)

	# Neural Networks does not understand strings

	# Bag of words training
	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		#bag of words
		bag = []

		stemmed_words = [stemmer.stem(w) for w in doc if w not in '?']

		for w in words:
			if w in stemmed_words:
				bag.append(1)
			else:
				bag.append(0)

			#copying the list	
			output_row = out_empty[:]
			output_row[labels.index(docs_y[x])] = 1

			training.append(bag)
			output.append(output_row)

	training = np.array(training)
	output = np.array(output)

	#wb = write bytes
	with open ("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)

#To reset the graph data
tf.reset_default_graph()

# defining the input data
net = tflearn.input_data(shape=[None, len(training[0])])


#2 hidden layers with 8 neurons 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

#connecting to the output that represents the patterns
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)


'''
THIS NEURAL NETWORKS

1-  The first layer represents the input (bag of words)
2 - Two layers of neurons
3 - The layer of the bag of words connects to each neuron (in the two layers) FULLY CONNECTED
4 - Output in the softmax layer activation 
5 - Return the probability for each neuron

For example: 

If the first neuron represents "hello", and the model
calculates the higher probability for "hello" to be the answer
'hello' will probably be the answer

Sorry for any mispelling, brazilian folk here :P
'''

model = tflearn.DNN(net)


# n_epoch => the number of times that the model will check the data
# show_metric => for visualization purposes
try:
	model.load("model.tflearn")
except:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")




def sentence_to_bag_of_words(sentence, li_words):
	# fill the list with zeros
	bag_wrds = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(sentence)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	#append 1 to specific parts
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag_wrds[i] = 1

	return np.array(bag_wrds)


def chat():
	print("Pergunte algo ao bot Frodo! Caso deseje sair, envie 'pare'.")
	while True:
		user_input = input(" Usuário: ")
		if user_input.lower() == 'pare' or 'parar':
			break

		#feed the model
		prediction = model.predict([sentence_to_bag_of_words(user_input, words)])
		#looking for the index of each word
		prediction_index = np.argmax(prediction)
		# for now returning the tag of the json file
		tag = labels[prediction_index]
		

		for tg in data['dataset']:
			if tg['tag'] == tag:
				responses = tg['responses']

		print(random.choice(responses))

chat()