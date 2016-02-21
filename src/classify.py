import os, random
import numpy as np
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from song_glob import SongGlob
import vis

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
    
def all_but_one(feature_list, glob, classifier=GaussianNB):
    all_features = glob.get_features(feature_list)
    all_classes = glob.get_feature('class')
    
    class_pred, class_real = [], []
    
    for i in range(len(all_features)):
        train_features = np.delete(all_features, i, 0)
        train_classes = np.delete(all_classes, i, 0)
        
        test_features = np.transpose(all_features[i,:]).reshape((1,11))
        test_class = np.transpose(all_classes[i,:])
        
        model = classifier()
        model.fit(train_features, train_classes.ravel())
        
        class_pred.append(test_model(test_features, model)[0])
        class_real.append(int_as_genre(test_class))
    
    return [class_pred, class_real, model]

if __name__ == '__main__':
    # load all into a glob
    glob = SongGlob()
    
    params_list = ['tempo', 'keystrength', 'eng', 'inharmonic', 'zerocross', 'chroma']
        
    # Gaussian Naive Bayes on tempo, keystrength, energy, inharmonicity
    p1, r1, gnb = all_but_one(params_list, glob)
    vis.present_results(p1, r1, "Gaussian Naive Bayes", True)

    # K-Nearest Neighbors on tempo and keystrength
    p2, r2, knn = all_but_one(params_list, glob, classifier=KNeighborsClassifier)
    vis.present_results(p2, r2, "Nearest Neighbors", True)
    
    p3, r3, sgd = all_but_one(params_list, glob, classifier=SGDClassifier)
    vis.present_results(p3, r3, "Stochastic Gradient Descent", True)
