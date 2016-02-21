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
    all_features = glob.get_features(feature_list, False)
    all_classes = glob.get_feature('class', False)
    print(len(all_features))
    class_pred, class_real = [], []
    
    for i in range(1000):
        train_features = all_features[:i] + all_features[(i+1):]
        train_classes = all_classes[:i] + all_classes[(i+1):]
        
        test_features = all_features[i]
        test_class = all_classes[i]
        
        model = classifier()
        model.fit(train_features, train_classes.ravel())
        
        class_pred.append(test_model(test_features, model))
        class_real.append(int_as_genre(test_class))
        
    return [class_pred, class_real, model]
    

def general(feature_list, glob, classifier=GaussianNB):
    # TODO: separate feature extraction and model training to optimize
    
    # compile feature vectors
    train_features, test_features = glob.get_features(feature_list, True)
    
    # grab the correct classifications
    train_classes, test_classes = glob.get_feature('class', True)

    # train
    model = classifier()
    model.fit(train_features, train_classes.ravel())

    # test
    class_pred = test_model(test_features, model)
    class_real = [int_as_genre(i) for i in test_classes]

    return [class_pred, class_real, model]

if __name__ == '__main__':
    # load all into a glob
    glob = SongGlob()
    glob.set_mask(train_fraction=.5)
    
    params_list = ['tempo', 'keystrength', 'eng', 'inharmonic', 'zerocross', 'chroma']
        
    # Gaussian Naive Bayes on tempo, keystrength, energy, inharmonicity
    p1, r1, gnb = general(params_list, glob)
    vis.present_results(p1, r1, "Gaussian Naive Bayes", True)

    # K-Nearest Neighbors on tempo and keystrength
    # p2, r2, knn = general(params_list, glob, classifier=KNeighborsClassifier)
    # vis.present_results(p2, r2, "Nearest Neighbors", True)
    # 
    # p3, r3, sgd = general(params_list, glob, classifier=SGDClassifier)
    # vis.present_results(p3, r3, "Stochastic Gradient Descent", True)
