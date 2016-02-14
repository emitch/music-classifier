import os, random, simple_classify, statistics
from sklearn.naive_bayes import GaussianNB
import numpy as np

def similarity_by_measure(tempo, time_series):
	""" break song up measure by measure, compare similarities between them
	for a given parameter, provided in time_series """
	# time per slice and per total song in seconds
	t_step = 0.02
	t_total = 30

	# find length of a measure in seconds
	beats_per_measure = 4
	t_measure = 60 * (beats_per_measure / tempo)

	# divide into measure-long chunks

	# TODO - finish that shit

def summarize(time_series):
	# take a long vector of junk and return only its mean and std
	n_rows = time_series.shape[0]
	stats = np.zeros([n_rows, 2])

	for i in range(n_rows):
		stats[i,0] = statistics.mean(time_series[i,:])
		stats[i,1] = statistics.stdev(time_series[i,:])

	return stats
