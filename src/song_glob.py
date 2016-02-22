import os, random, sys
import scipy.io as sio
import scipy.stats.stats as st
import numpy as np
import matplotlib.pyplot as plt

class SongGlob:
    def load_songs(self, data_folder='../data'):
        # load in data as list of matlab structs

        # initialize empty list
        songs = []
        song_count = 0
        # walk through data folder, touching each subfolder
        for path, dirs, files in os.walk(data_folder):
            # loop through files in subfolder
            for file in files:
                # find .mat files
                if file.endswith('.mat'):
                    # load only data from files, using corresponding path
                    # add to list of song data
                    songs.append(sio.loadmat(path + '/' + file)['DAT'])
                    
                    song_count += 1
                    sys.stdout.write("\rLoaded %d / 1000 songs" % song_count)

        print("")
        
        return songs

    def __init__(self):
        self.data = self.load_songs()
    
    # gets a matrix of the requested features
    def get_features(self, feature_list, include_dominant=False):
        for feature in feature_list:
            result = self.get_feature(feature)
            
            feature_matrix = np.empty([0,0])
            if len(result[0]) > 0:
                if 'feature_matrix' not in locals():
                    feature_matrix = result
                else:
                    feature_matrix = np.hstack((feature_matrix, result))
        
        if include_dominant:
            for feature in feature_list:
                result = self.get_feature(feature, True)
                
                if len(result[0]) > 0:
                    feature_matrix = np.hstack((feature_matrix, result))
        
        return feature_matrix

    # gets a list of only one specific feature
    def get_feature(self, feature_name, dominant_key=False):
        # extract a given parameter name from the big array
        
        # since the vectors aren't all the same size, get min dimensions
        n_songs = len(self.data)
        n_rows = self.data[0][feature_name][0][0].shape[0]
        n_cols = self.data[0][feature_name][0][0].shape[1]

        # single element parameters are easy
        if n_cols == 1:
            if dominant_key:
                return [np.array([]), np.array([])]
                
            # grab each item and transpose to column vector
            grabbed = np.vstack(
                [self.data[i][feature_name][0][0][0][0] 
                for i in range(len(self.data))])
        elif n_rows == 12:
            # select only the row corresponding to the key of the song
            # get a matrix of features one row at a time
            summaries = []
            for i in range(n_songs):
                # get key, only use info in relevant key
                key = self.data[i]['key'][0][0][0][0]
                
                if dominant_key:
                    key = key + 7
                    if key > 12:
                        key = key - 12
                
                # mean and std
                mean = np.nanmean(self.data[i][feature_name][0][0][key-1,:])
                std = np.nanstd(self.data[i][feature_name][0][0][key-1,:])
                
                # add to running list
                summaries.append(np.array([mean, std]))

            grabbed = np.vstack(summaries)
        elif n_rows == 24:
            raise

        else:
            if dominant_key:
                return [np.array([]), np.array([])]
                
            # simply summarize the row
            summaries = []
            for i in range(n_songs):
                mean = np.nanmean(self.data[i][feature_name][0][0][0,:])
                std = np.nanstd(self.data[i][feature_name][0][0][0,:])
                
                summaries.append(np.array([mean, std]))

            grabbed = np.vstack(summaries)

        return grabbed

    def plot_feature(self, feature_name, plot_all=False):
        n_classes = 10

        colors = ['dimgray', 'red', 'brown', 'darkgray', 'yellow', 'palegreen', 
            'seagreen', 'deepskyblue', 'navy', 'deeppink']
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
            'jazz', 'metal', 'pop', 'reggae', 'rock']

        # get a matrix of the required features, along with class
        feature_vector = self.get_feature(feature_name)
        classes = self.get_feature('class')

        if feature_vector.shape[1] != 2:
            raise


        # plot them by class
        plt.figure(1)
        plot_handles = []
        for i in range(1,n_classes + 1):
            # extract by class
            subset = feature_vector[classes[:,0]==i,:]
            if plot_all:
                # plot all in a given class in the same color
                plot_handles.append(
                    plt.scatter(subset[:,0], subset[:,1], c = colors[i-1]))
            else:
                # plot the mean of each class and use std for error bars
                plot_handles.append(
                    plt.errorbar(np.nanmean(subset[:,0]), np.nanmean(subset[:,1]),
                    xerr = np.nanstd(subset[:,0]), yerr = np.nanstd(subset[:,1]),
                    mfc = colors[i-1], mec = colors[i-1], ecolor = colors[i-1],
                    fmt = 'o'))

        plt.title(feature_name + ' statistics by genre')
        plt.xlabel('Mean of parameter')
        plt.ylabel('Standard deviation of parameter')
        plt.legend(plot_handles, [genres[j] for j in range(n_classes)])
        plt.show()
