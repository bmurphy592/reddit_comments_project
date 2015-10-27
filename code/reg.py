# Python script for predicting the scores of comments

import numpy as np 
import sqlite3

conn = sqlite3.connect('../data/database.sqlite')
train_size = 1

def load_training_data():

	# The training set. Selects train_size entries. May want to select specific columns in the future.
	# Train is a list of tuples where each tuple is #columns long
	train = conn.execute('SELECT * FROM May2015 LIMIT '+str(train_size))

	for row in train:
		print row

def train():
	pass

if __name__=='__main__':
	load_training_data()
