import classify, vis, os, random
import numpy as np
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def test_model(data, model):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']

    # only predict genres for those not used by mask
    predictions = model.predict(data)

    # convert to words
    predictions = [genres[int(genre)-1] for genre in predictions]

    return predictions

def make_mask(n_total, n_mask):
    # select training set using boolean mask of a random subset
    training_indices = random.sample(range(n_total), n_mask)

    mask = np.zeros(n_total, dtype=bool)
    mask[training_indices] = True

    return mask

def int_as_genre(number):
    # take an integer and return its string genre
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    return genres[number-1]

def grab_param(data, param):
    # extract a given parameter name from the big array

    # since the vectors aren't all the same size, get min dimensions
    n_songs = len(data)
    n_rows = data[0][param][0][0].shape[0]
    n_cols = data[0][param][0][0].shape[1]

    # single element parameters are easy
    if n_cols == 1:
        # grab each item and transpose to column vector
        grabbed = np.vstack(
            [data[i][param][0][0][0][0] for i in range(len(data))])
    elif n_rows == 12:
        # select only the row corresponding to the key of the song
        # get a matrix of features one row at a time
        summaries = []
        for i in range(n_songs):
            # get key, only use info in relevant key
            key = data[i]['key'][0][0][0][0]
            # mean and std
            mean = np.nanmean(data[i][param][0][0][key-1,:])
            std = np.nanstd(data[i][param][0][0][key-1,:])
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
                [np.nanmean(data[i][param][0][0][0,:]), 
                np.nanstd(data[i][param][0][0][0,:])]))

        grabbed = np.vstack(summaries)

    return grabbed

def summarize(time_series):
    # take a long vector of junk and return only its mean and std
    n_rows = time_series.shape[0]
    stats = np.zeros([n_rows, 2])

    for i in range(n_rows):
        stats[i,0] = np.nanmean(time_series[i,:])
        stats[i,1] = np.nanstd(time_series[i,:])

    return stats

def general(params_list, data, training_fraction=0.5,
    classifier=GaussianNB):
    # TODO: separate feature extraction and model training to optimize
    
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
    class_pred = test_model(params[~mask,:], model)
    class_real = [int_as_genre(i) for i in classes[~mask]]

    return [class_pred, class_real, model, mask]


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

    # Gaussian Naive Bayes on tempo, keystrength, energy, inharmonicity
    p1, r1, gnb, m1 = general(
        ['tempo', 'keystrength', 'eng', 'inharmonic'], data)
    vis.present_results(p1,r1)

    # K-Nearest Neighbors on tempo and keystrength
    p2, r2, knn, m2 = general(
        ['tempo', 'keystrength', 'eng', 'inharmonic'], data, 
        classifier=KNeighborsClassifier)
    vis.present_results(p2,r2)



