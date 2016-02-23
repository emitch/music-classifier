import os, random, simple_classify, statistics
from sklearn.naive_bayes import GaussianNB
from scipy import signal
import numpy as np

# constants
SAMPLING_RATE = 44100
t_STEP = 0.02
BEATS_PER_MEASURE = 4

def separate_measures(tempo, time_series):
	""" break song up measure by measure, compare similarities between them
	for a given parameter, provided in time_series (a 2D numpy array) """
	# time per slice and per total song in seconds
	t_total = time_series.shape[1] * t_STEP

	# find length of a measure in seconds
	t_measure = 60 * (BEATS_PER_MEASURE / tempo)

	# get real time at beginning of each measure-long chunk
	measure_beginnings = np.arange(0, t_total - t_measure, t_measure)
	n_measures = len(measure_beginnings)
	# number of non-conflicting frames belonging to each measure
	measure_length = int(round(t_measure / t_STEP))

	# gather each measure and stack them on top of each other in a 3D array
	measures = np.empty([time_series.shape[0], measure_length, n_measures])
	for i in range(n_measures):
		# closest index to which measure i begins (rounded)
		idx =  int(round(measure_beginnings[i] / t_STEP))
		# fill in proper slice of measure vector
		measures[:,:,i] = time_series[:, idx:idx + measure_length]

	return measures

def compare_measures(measures, squeeze = False):
	""" take a 3D numpy array of features organized by measure and return
	the mean of the variance of each measure-normalized frame """
	# compute variance along the 3rd dimension, which is measure number
	variance = np.nanvar(measures, 2)
	# normalize by the mean of the same feature
	# compute the mean of these scores along the second dimension
	score = np.nanmean(variance, 1)

	# if squeeze, then squeeze the whole thing down to one scalar
	if squeeze:
		return np.sum(score * score)

	# return a vector of scores, one for each row of the input
	return score

def raw_feature(song, feature):
	feature = song[feature][0][0]
	if feature.shape == (1,1):
		return feature[0][0]
	else:
		return feature

def feat_by_measure(song, feature):
	separated = separate_measures(raw_feature(song, 'tempo'), 
		raw_feature(song, feature))
	score = compare_measures(separated, True)
	return score

def separate_beats():
	""" just like separate measures but do it for each beat instead """
	return

# never actually used this, but it was a nice thought
def wavelet(raw):
	""" DO WAVELET """
	# numbers of samples over which to compute wavelet
	widths = np.logspace(1,12,num=10,base=2) # freq of each w is (44100/w) Hz
	# compute wavelet transform
	wvlt = signal.cwt(raw, signal.ricker, widths)
	# TODO: use detected bpm as a width for wavelets