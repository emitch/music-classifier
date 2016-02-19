import classify, vis, os, random
import numpy as np
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from song_glob import SongGlob

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

def general(feature_list, glob, classifier=GaussianNB):
    # TODO: separate feature extraction and model training to optimize
    
    # compile feature vectors
    train_features, test_features = glob.get_features(feature_list, True)

    # grab the correct classifications
    train_classes, test_classes = glob.get_feature('class', True)

    # train
    model = classifier()
    model.fit(train_features, train_classes)

    # test
    class_pred = test_model(test_features, model)
    class_real = [int_as_genre(i) for i in test_classes]

    return [class_pred, class_real, model]

if __name__ == '__main__':
    # load all
    glob = SongGlob()
    glob.set_mask(train_fraction=.5)
    
    # change to true to run a loop to only see overall accuracy with different
    # training sizes
    RUN_LOOP = False

    if RUN_LOOP:
        train_fraction = .01
        # Gaussian Naive Bayes on tempo, keystrength, energy, inharmonicity
        for i in range(100):
            print("fraction:", train_fraction)
            p1, r1, gnb = general(
                ['tempo', 'keystrength', 'eng', 'inharmonic'], glob)
            vis.present_results(p1,r1,True)
            glob.set_mask(train_fraction)
            train_fraction += .01
        
    # Gaussian Naive Bayes on tempo, keystrength, energy, inharmonicity
    p1, r1, gnb = general(
        ['tempo', 'keystrength', 'eng', 'inharmonic'], glob)
    vis.present_results(p1,r1,False)

    # K-Nearest Neighbors on tempo and keystrength
    # p2, r2, knn = general(
    #     ['tempo', 'keystrength', 'eng', 'inharmonic'], glob, 
    #     classifier=KNeighborsClassifier)
    # vis.present_results(p2,r2)
