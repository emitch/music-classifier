import os, random, simple_classify, statistics
from sklearn.naive_bayes import GaussianNB
import numpy as np

# global variables
FRAMES_PER_SONG = 1198

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



