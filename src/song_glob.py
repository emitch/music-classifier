import os, random, sys, features
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class SongGlob:
    def load_songs(self, data_folder='../data'):
        # load in data as list of matlab structs

        # initialize empty list
        songs = []
        song_count = 0
        idx = 0
        # walk through data folder, touching each subfolder
        for path, dirs, files in os.walk(data_folder):
            # loop through files in subfolder
            for file in files:
                # find .mat files
                if file.endswith('.mat'):
                    if idx % 100 < 100:
                        # load only data from files, using corresponding path
                        # add to list of song data
                        songs.append(sio.loadmat(path + '/' + file)['DAT'])
                        
                        song_count += 1
                        sys.stdout.write("\rLoaded %d songs" % song_count)
                        
                    idx += 1

        print("")
        
        return songs

    def __init__(self):
        self.data = self.load_songs()
        self.n = len(self.data)
    
    # gets a matrix of the requested features
    def get_features(self, feature_dict):
        for feature in feature_dict:
            result = self.get_feature(feature, feature_dict[feature])
            
            if len(result[0]) > 0:
                if 'feature_matrix' not in locals():
                    feature_matrix = result
                else:
                    feature_matrix = np.hstack((feature_matrix, result))
        
        return feature_matrix

    # gets a list of only one specific feature
    def get_feature(self, feature_name, feature_tup):
        get_mean = feature_tup[0]
        get_std = feature_tup[1]
        get_measure = feature_tup[2]
        
        # extract a given parameter name from the big array
        
        # since the vectors aren't all the same size, get min dimensions
        n_songs = len(self.data)
        n_rows = self.data[0][feature_name][0][0].shape[0]
        n_cols = self.data[0][feature_name][0][0].shape[1]

        # single element parameters are easy
        if n_cols == 1:
            # grab each item and transpose to column vector
            grabbed = np.vstack(
                [self.data[i][feature_name][0][0][0][0] 
                for i in range(len(self.data))])

        elif n_rows == 12:
            # select only the row corresponding to the key of the song
            # get a matrix of features one row at a time
            summaries = []
            for i in range(n_songs):
                fv = []
                # get key, only use info in relevant key
                key = self.data[i]['key'][0][0][0][0]
                dominant_key = key + 7
                if dominant_key > 12:
                    dominant_key = dominant_key - 12
                
                # mean and std
                if get_mean:
                    fv.append(np.nanmean(self.data[i][feature_name][0][0][key-1,:]))
                    fv.append(np.nanmean(self.data[i][feature_name][0][0][dominant_key-1,:]))
                if get_std:
                    fv.append(np.nanstd(self.data[i][feature_name][0][0][key-1,:]))
                    fv.append(np.nanstd(self.data[i][feature_name][0][0][dominant_key-1,:]))
                if get_measure:
                    fv.append(features.feat_by_measure(self.data[i], feature_name))
                
                # add to running list
                summaries.append(np.array(fv))

            grabbed = np.vstack(summaries)

        else:
            # simply summarize the first (perhaps only) row 
            summaries = []
            for i in range(n_songs):
                fv = []
                if get_mean:
                    fv.append(np.nanmean(self.data[i][feature_name][0][0][0,:]))
                if get_std:
                    fv.append(np.nanstd(self.data[i][feature_name][0][0][0,:]))
                if get_measure:
                    fv.append(features.feat_by_measure(self.data[i], feature_name))

                summaries.append(np.array(fv))

            grabbed = np.vstack(summaries)

        return grabbed

    def plot_feature(self, feature_name, plot_all=False):
        n_classes = 10

        colors = ['dimgray', 'red', 'brown', 'darkgray', 'yellow', 'palegreen', 
            'seagreen', 'deepskyblue', 'navy', 'deeppink']
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 
            'metal', 'pop', 'reggae', 'rock']

        # get column vector of class assignments
        classes = self.get_feature('class')

        # for basic features w/string name use get_feature
        if isinstance(feature_name, str):
            # get a matrix of the required features
            feature_vector = self.get_feature(feature_name)
        else:
            # assume feature is a function on a single entry of glob.data
            # returning a 1x2 numpy arrray
            features = []
            for i in range(len(self.data)):
                features.append(feature_name(self.data[i]))

            feature_vector = np.vstack(features)

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

        plt.title('2D Feature space by genre')
        plt.xlabel('Parameter 1')
        plt.ylabel('Parameter 2')
        plt.legend(plot_handles, [genres[j] for j in range(n_classes)])
        plt.show()
