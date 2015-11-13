# Python script for predicting the scores of comments

import numpy as np 
import sqlite3
import pdb
import random
import string
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt

conn = sqlite3.connect('../data/database.sqlite')
data_size = 1000

def load_data():

	cursor = conn.execute("SELECT created_utc, gilded, author, body, controversiality, edited, score FROM May2015 WHERE subreddit = 'aww' LIMIT "+str(data_size));
	data = cursor.fetchall()
	random.shuffle(data)

	user_count = 0
	words_count = 0
	def neg_one(): return -1
	users = defaultdict(neg_one)
	words = defaultdict(neg_one)
	samples = len(data)
	print 'Converting users'
	for i in range(samples):
		entry = data[i]
		exclude = set(string.punctuation)
		body = ''.join(ch for ch in entry[3] if ch not in exclude).lower()
		tokens = body.split()
		if users[entry[2]] < 0:
			users[entry[2]] = user_count
			user_count += 1

		for token in tokens:
			if words[token] < 0:
				words[token] = words_count
				words_count += 1
		entry = list(entry)
		entry[3] = body
		data[i] = entry

	print 'Converting body'
	for i in range(samples):
		word_vec = [0] * len(words.keys())
		user_vec = [0] * len(users.keys())
		for token in data[i][3].split():
			word_vec[words[token]] += 1
		user_vec[users[data[i][2]]] = 1
		del data[i][2]
		del data[i][2]
		data[i][:0] = user_vec
		data[i][:0] = word_vec
		# pdb.set_trace()

	data = np.array(data)
	scores = data[:,-1]
	data = data[:, :-1]
	# pdb.set_trace()
	data = data - np.tile(data.min(axis=0), (samples, 1))
	maxes = data.max(axis=0)
	maxes[maxes == 0] = 1
	data = data / maxes
	# pdb.set_trace()
	train = data[(len(data)/2):, :-1]
	test = data[:(len(data)/2), :-1]
	train_scores = scores[(len(data)/2):]
	test_scores = scores[:(len(data)/2)]

	print 'Created train and test sets'
	return train, test, train_scores, test_scores

def train(train_set, scores):
	print 'Training model'
	# pdb.set_trace()
	# model = Ridge().fit(train_set, scores)
	model = SVR(kernel = 'rbf').fit(train_set, scores)
	print 'Finished training model'
	return model

def test(model, test_set, scores):
	print 'Computing generalization error'
	return model.score(test_set, scores), model.predict(test_set)

def plot(predictions, actual):
	samples = range(0, len(predictions))
	plt.plot(samples, predictions, 'ro')
	plt.plot(samples, actual, 'bo')
	plt.axis([0, samples[-1], -20, 20])
	plt.show()

if __name__=='__main__':
	# pdb.set_trace()
	train_set, test_set, train_scores, test_scores = load_data()
	model = train(train_set, train_scores)
	score, predictions = test(model, test_set, test_scores)
	# plot(predictions, test_scores)
	pdb.set_trace()

