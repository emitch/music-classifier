import os, random
from sklearn.naive_bayes import GaussianNB
import numpy as np
import scipy.io as sio
import vis

def build_model(params, data):
    """ 
    take an array of parameter values and the corresponding observed 
    data and return a gaussian naive bayes model to represent the data
    """
    # initialize model
    model = GaussianNB()
    # fit model (woah so simple!)
    model.fit(params, data)

    return model

def extract_params(param_names, structure):
    """
    param_names is a list of field names found in structure, this will
    extract only the values from the requested fields and concetenate them
    into a single numpy array that can be passed to classification methods
    """
    # create numpy array
    array_length = 0
    n_trials = len(structure)
    # figure out how large to make array
    for param in param_names:
        # structure is frustratingly complicated w/nested arrays
        array_length += structure[0][param][0][0].size

    extracted = np.empty([n_trials, array_length])

    # loop through each parameter name and get it from structure
    for i in range(n_trials):
        entry = np.empty([0])
        for param in param_names:
            # get field of structure, should work sorta like a dictionary
            arr = structure[i][param][0][0]
            # reshape and add to cumulative structure
            # final version will be a matrix, each column a parameter,
            # each row a single observation
            entry = np.append(entry, np.reshape(arr, [1, arr.size]))
        # add row to big structure
        extracted[i,:] = entry

    return extracted

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
                if random.randrange(10) == 0:
                    songs.append(sio.loadmat(path + '/' + file)['DAT'])

    return songs

def genre_as_int(genre):
    # take a genre name and return its integer classifier
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']

    return genres.index(genre)

def int_as_genre(number):
    # take an integer and return its string genre
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    return genres[number-1]

# run a shitty classification using only tempo and key
if __name__ == '__main__':
    # ** TRAIN **

    # load in songs
    data_raw = load_songs()

    # trim to only key and tempo for each song
    data_trimmed = extract_params(['key', 'tempo'], data_raw)
    data_genres = extract_params(['class'], data_raw)

    # select training set using boolean mask of a random subset
    training_indices = random.sample(range(data_trimmed.shape[0]), data_trimmed.shape[0]//2)

    mask = np.zeros(data_trimmed.shape[0], dtype=bool)
    mask[training_indices] = True

    training_params = data_trimmed[mask,:]
    training_values = data_genres[mask]

    # build model
    model = build_model(training_params, training_values)

    # ** TEST **

    # only predict genres for those not used by mask
    guessed_genres = model.predict(data_trimmed[~mask])
    correct_genres = data_genres[~mask]

    guessed_genres = [int_as_genre(int(genre)) for genre in guessed_genres]
    correct_genres = [int_as_genre(int(genre)) for genre in correct_genres]

    print(data_raw[0]['keystrength'][0][0])

    # evaluate and graph
    vis.present_results(guessed_genres, correct_genres)
