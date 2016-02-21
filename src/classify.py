import os, random
import numpy as np
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from song_glob import SongGlob
import vis
from itertools import chain, combinations
import sys
import time

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable([combinations(s, r) for r in range(len(s)+1)])

def test_model(data, model):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']

    # only predict genres for those not used by mask
    predictions = model.predict(data)

    # convert to words
    predictions = [genres[int(genre)-1] for genre in predictions]

    return predictions

def int_as_genre(number):
    # take an integer and return its string genre
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
        'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    return genres[number-1]
    
def leave_one_out(feature_list, glob, classifier=GaussianNB):
    all_features = glob.get_features(feature_list)
    all_classes = glob.get_feature('class')
    
    class_pred, class_real = [], []
    
    for i in range(len(all_features)):
        train_features = np.delete(all_features, i, 0)
        train_classes = np.delete(all_classes, i, 0)
        
        test_features = np.transpose(all_features[i,:]).reshape((1, train_features.shape[1]))
        test_class = np.transpose(all_classes[i,:])
        
        model = classifier()
        model.fit(train_features, train_classes.ravel())
        
        class_pred.append(test_model(test_features, model)[0])
        class_real.append(int_as_genre(test_class))
    
    return [class_pred, class_real, model]

if __name__ == '__main__':
    all_features = ['eng', 'chroma', 't', 'keystrength', 'brightness', 'zerocross', 'roughness', 'inharmonic', 'tempo', 'key']
    all_features = list(powerset(all_features))
    random.shuffle(all_features)
    
    # load all into a glob
    glob = SongGlob()
    
    idx = 0
    results = {}
    
    start = time.clock()
    
    for feature_set in all_features:
        if len(feature_set) == 0:
            continue
        
        p1, r1, gnb = leave_one_out(feature_set, glob)
        acc = vis.present_results(p1, r1, "Gaussian Naive Bayes", print_results=False, show_results=False)
        
        results[feature_set] = acc
        
        idx += 1
        
        t = time.clock() - start
        
        remaining = t * (len(all_features) / idx) - t
        
        sys.stdout.write("\r%d / %d permutations (%.2f minutes left)" % (idx, len(all_features), remaining / 60))
    
    for set in sorted(results, key=results.get, reverse=True):
        print(set, results[set])
    
    params_gnb = ['tempo', 'keystrength', 'eng', 'inharmonic', 'zerocross', 'chroma']
    params_knn = ['inharmonic']
    params_sgd = ['tempo', 'keystrength', 'eng', 'inharmonic', 'zerocross', 'chroma']
        
    # Gaussian Naive Bayes on tempo, keystrength, energy, inharmonicity
    p1, r1, gnb = leave_one_out(params_gnb, glob)
    vis.present_results(p1, r1, "Gaussian Naive Bayes", print_results=False, show_results=False)

    # K-Nearest Neighbors on tempo and keystrength
    p2, r2, knn = leave_one_out(params_knn, glob, classifier=KNeighborsClassifier)
    vis.present_results(p2, r2, "Nearest Neighbors", print_results=False, show_results=False)
    
    p3, r3, sgd = leave_one_out(params_sgd, glob, classifier=SGDClassifier)
    vis.present_results(p3, r3, "Stochastic Gradient Descent", print_results=False, show_results=False)
