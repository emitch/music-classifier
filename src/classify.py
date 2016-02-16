import classify, vis, os, random, statistics
import numpy as np
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB

# global variables
FRAMES_PER_SONG = 1198

def test_model(data, model):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']

    # only predict genres for those not used by mask
    predictions = model.predict(data)

    # convert to words
    predictions = [genres[int(genre)-1] for genre in guessed_genres]

    return predictions

def make_mask(n_total, n_mask):
    # select training set using boolean mask of a random subset
    training_indices = random.sample(range(n_total), n_mask)

    mask = np.zeros(n_total, dtype=bool)
    mask[training_indices] = True

    return mask

def grab_param(data, param, summarize_flag=True, key_only=True):
    # extract a given parameter name from the big array

    # since the vectors aren't all the same size, get min dimensions
    n_songs = len(data)
    n_rows = min([data[i][param][0][0].shape[0] for i in range(n_songs)])
    n_cols = min([data[i][param][0][0].shape[1] for i in range(n_songs)])

    # single element parameters are easy
    if n_cols == 1:
        # grab each item and transpose to column vector
        grabbed = np.vstack(
            [data[i][param][0][0][0][0] for i in range(len(data))])
    else:
        grabbed = np.empty((len(data), 0))
        # use the size and loop through
        if summarize_flag:
            # TODO: select only the row corresponding to the key of the song
            for row in range(n_rows):
                # get a matrix of features one row at a time
                this_row = np.vstack([np.array([
                    statistics.mean(data[i][param][0][0][row,0:n_cols]), 
                    statistics.stdev(data[i][param][0][0][row,0:n_cols])])
                    for i in range(n_songs)])

                # summarize it and add to grabbed
                grabbed = np.hstack([grabbed, this_row])
        else:
            # not written yet
            raise

    return grabbed

def summarize(time_series):
    # take a long vector of junk and return only its mean and std
    n_rows = time_series.shape[0]
    stats = np.zeros([n_rows, 2])

    for i in range(n_rows):
        stats[i,0] = statistics.mean(time_series[i,:])
        stats[i,1] = statistics.stdev(time_series[i,:])

    return stats

def classify_1(data):
    # extract strength of dominant key
    key_strength = np.zeros((len(data), FRAMES_PER_SONG))
    for i in range(len(data)):
        key = data[i]['key'][0][0][0][0]
        key_strength[i,:] = data[i]['keystrength'][0][0][key-1,:]

    # extract only mean + std
    key_strength = classify.summarize(key_strength)

    # extract tempo
    tempo = grab_param(data, 'tempo')
    # combine arrays:
    params = np.hstack((key_strength, tempo))
    # get correct genre assignment
    classes = grab_param(data, 'class')
    # select random subsets for training + testing
    mask = make_mask(params.shape[0], params.shape[0]//2)

    # train
    model = GaussianNB()
    model.fit(params[mask,:], classes[mask])

    # test
    return [test_model(data[~mask,:], model), classes[~mask]]

def classify_general(params_list, data, training_fraction=0.5,
    classifier=GaussianNB):

    # compile feature vectors
    params = np.hstack([grab_param(data, param) for param in params_list])

    # grab the correct classifications
    classes = grab_param(data, 'class')

    # make mask
    mask = make_mask(len(data), int(training_fraction*len(data)))

    # train
    model = classifier()
    model.fit(params[mask,:], classes[mask])

    # test
    return [test_model(params[~mask,:], model), classes[~mask]]


def load_songs(data_folder = '../data'):
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

if __name__ == '__main__':
    # load all
    data = load_songs()

    # do some classifying
