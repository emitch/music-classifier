import os, random
import scipy.io as sio
import numpy as np

class SongGlob:
    def load_songs(self, data_folder='../data'):
        # load in data as list of matlab structs

        # initialize empty list
        songs = []
        # walk through data folder, touching each subfolder
        for path, dirs, files in os.walk(data_folder):
            # loop through files in subfolder
            for file in files:
                # find .mat files
                if file.endswith('.mat'):
                    # load only data from files, using corresponding path
                    # add to list of song data
                    songs.append(sio.loadmat(path + '/' + file)['DAT'])

        return songs

    def __init__(self):
        self.data = self.load_songs()
        
    def set_mask(self, train_fraction=.5):
        n_total = len(self.data)
        n_train = int(n_total * train_fraction)
        
        # select training set using boolean mask of a random subset
        training_indices = random.sample(range(n_total), n_train)
        
        m = np.zeros(n_total, dtype=bool)
        m[training_indices] = True
        
        self.mask = m
        
    def get_features(self, feature_list, mask_split):
        for feature in feature_list:
            train, test = self.get_feature(feature, mask_split)
            if 'train_features' not in locals():
                train_features = train
                test_features = test
            else:
                train_features = np.hstack((train_features, train))
                test_features = np.hstack((test_features, test))
        
        return (train_features, test_features)

    def get_feature(self, feature_name, mask_split):
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
                # get key, only use info in relevant key
                key = self.data[i]['key'][0][0][0][0]
                # mean and std
                mean = np.nanmean(self.data[i][feature_name][0][0][key-1,:])
                std = np.nanstd(self.data[i][feature_name][0][0][key-1,:])
                # add to running list
                summaries.append(np.array([mean, std]))

            grabbed = np.vstack(summaries)
        elif n_rows == 24:
            raise

        else:
            # simply summarize the row
            summaries = []
            for i in range(n_songs):
                summaries.append(np.array(
                    [np.nanmean(self.data[i][feature_name][0][0][0,:]), 
                    np.nanstd(self.data[i][feature_name][0][0][0,:])]))

            grabbed = np.vstack(summaries)
        
        if mask_split:
            return (grabbed[self.mask], grabbed[~self.mask])
        else:
            return grabbed
        
