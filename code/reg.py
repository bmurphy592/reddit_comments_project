# Python script for predicting the scores of comments
from __future__ import division
import numpy as np 
import sqlite3
import pdb
import random
import string
import math
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

conn = sqlite3.connect('../data/database.sqlite')
data_size = 100

# Calculates the entropy of a comment's body
def calc_entropy(word_array):
	entropy = 0
	num_words = len(word_array)
	frequencies = defaultdict(int)
	for word in word_array:
		frequencies[word] += 1
	for word in word_array:
		entropy += (frequencies[word] / num_words) * (np.log10(num_words) - np.log10(frequencies[word]))
	entropy *= (1/num_words)
	return entropy

# Strips punctuation and converts comment body to array of words
def convert_to_word_array(body):
	exclude = set(string.punctuation)
	body = ''.join(ch for ch in body if ch not in exclude).lower()
	tokens = body.split()
	return tokens

# Returns a dictionary of usernames -> number of comments by that user
def get_num_comments(data):
	num_comments = defaultdict(int)
	for entry in data:
		num_comments[entry[2]] += 1
	return num_comments

def inverse_document_frequency(word, documents):
	count = sum(1 for d in documents if word in d)
	return math.log(len(documents) / count)

def tf_idf(word, document, documents):
	word_count = sum(1 for w in document if w == word)
	tf = word_count / len(document)
	idf = inverse_document_frequency(word, documents)
	return tf * idf

# Returns a class label from a comment's score:
# 0 = score <-10
# 1 = -10 <= score <= 0
# 2 = 0 < score < 10
# 3 = score >= 10
def get_class_label(score):
	if score < -10:
		return 0
	elif score <= 0:
		return 1
	elif score < 10:
		return 2
	return 3

# Modifies the data such that every feature value is between 0 and 1 inclusive
def normalize(data):
	data = data - np.tile(data.min(axis=0), (len(data), 1))
	maxes = data.max(axis=0)
	maxes[maxes == 0] = 1
	data = data / maxes
	return data

def load_data(subreddit):
	cursor = conn.execute("SELECT created_utc, gilded, author, body, controversiality, score FROM May2015 WHERE subreddit = \'"+subreddit+"\' LIMIT "+str(data_size));
	data = cursor.fetchall()
	random.shuffle(data)
	samples = len(data)

	num_comments_dict = get_num_comments(data)

	

	for i in range(samples):
		entry = data[i]
		body = convert_to_word_array(entry[3])
		entropy = calc_entropy(body) if len(body) > 0 else 0
		comment_length = len(body)
		num_comments = num_comments_dict[entry[2]]
		entry = list(entry)
		del entry[2]
		del entry[2]
		entry = [entropy, comment_length, num_comments] + entry
		entry[-1] = get_class_label(entry[-1])
		data[i] = entry

	data = np.array(data)
	scores = data[:,-1]
	data = data[:, :-1]
	data = normalize(data)
	train = data[(samples/2):, :-1]
	test = data[:(samples/2), :-1]
	train_scores = scores[(samples/2):]
	test_scores = scores[:(samples/2)]

	print 'Created train and test sets'
	return train, test, train_scores, test_scores

def exploration():
	cursor = conn.execute("SELECT body FROM May2015 WHERE subreddit = 'gaming' LIMIT "+str(data_size));
	data = cursor.fetchall()
	samples = len(data)
	data = [convert_to_word_array(body[0]) for body in data]
	idfs = {}
	words = set()
	for body in data:
		for w in body:
			if w not in words:
				print 'Computing frequency of', w
				idfs[w] = inverse_document_frequency(w, data)
				words = words.union([w])
	results = sorted(idfs.items(), key=lambda x: (-x[1], x[0]))
	# 	results = defaultdict(int)
	# 	for w in body:
	# 		print "Calculating for", w
	# 		results[w] = tf_idf(w, body, data)
	# 		words = words.union([w])
	# 	tf_idfs.append(results)
	pdb.set_trace()

# Parameters:
# subreddit - the subreddit that is being trained/tested on
# percentage - the top X percentage of words to be included
# Output:
# List of top percentage words with the highest idf
def important_words(subreddit, percentage):
	cursor = conn.execute("SELECT body FROM May2015 WHERE subreddit = \'"+subreddit+"\' LIMIT "+str(data_size));
	data = cursor.fetchall()
	samples = len(data)
	data = [convert_to_word_array(body[0]) for body in data]
	idfs = {}
	for body in data:
		for w in set(body) - set(idfs.keys()):
			print 'Computing frequency of', w
			idfs[w] = inverse_document_frequency(w, data)
	results = sorted(idfs.items(), key=lambda x: (-x[1], x[0]))
	best_words = [x[0] for x in results[:int(math.floor(len(results)*percentage))]]
	return best_words


def train(train_set, scores):
	print 'Training model'
	model = svm.SVC(kernel='rbf').fit(train_set, scores)
	print 'Finished training model'
	return model

def test(model, test_set, scores):
	print 'Computing generalization error'
	return model.score(test_set, scores), model.predict(test_set)

def plot(predictions, actual, data):
	gx1, gx2 = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
	labels = ['.r', '.g', '.b', '.y']
	c_map = ListedColormap(['#FFAAAA', '#99FF99', '#AAAAFF', '#FFFF99'])
	for i in range(len(actual)):
		plt.plot(data[i][0], data[i][1], labels[np.int_(actual[i])], alpha=0.5)
	pdb.set_trace()
	predictions = predictions.reshape(gx1.shape)
	plt.pcolormesh(gx1, gx2, predictions, cmap=c_map)
	plt.show()

if __name__=='__main__':
	random.seed(1000)
	important_words('gaming', .75)
	# train_set, test_set, train_scores, test_scores = load_data()
	# model = train(train_set, train_scores)
	# score, predictions = test(model, test_set, test_scores)
	# # plot(predictions, test_scores, test_set)
	# pdb.set_trace()
	# print score

