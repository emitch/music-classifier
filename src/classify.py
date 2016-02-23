import os, random, sys, time, copy
import numpy as np
import scipy.io as sio
from itertools import chain, combinations

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# our own stuff
from song_glob import SongGlob
import vis

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
    'jazz', 'metal', 'pop', 'reggae', 'rock']

# helper function to build a powerset from an iterable type
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable([combinations(s, r) for r in range(len(s)+1)])

# convert an integer genre to its text form
def genre_from_int(number):
    return genres[number-1]

# convert a genre string to an int
def int_from_genre(genre):
    return genres.index(genre)

##################################
# SUPPORT VECTOR MACHINE FUNCTIONS
################

# build a gram matrix for a given set of train vectors and test vectors
def gram(train_features, test_features):
    gram = np.empty([len(test_features), len(train_features)])
    
    for i in range(gram.shape[0]):
        for j in range(gram.shape[1]):
            gram[i, j] = kernel(test_features[i], train_features[j])
    
    return gram

# compute the similarity of two feature vectors
#
# we define the kernel as the negative sum of the difference between the
# features, with 0 being identical objects
def kernel(x1, x2):
    similarity = 0
    for idx in range(len(x1)):
        if x1[idx] != x2[idx]:
            similarity -= abs(x1[idx] - x2[idx])
    
    return similarity
    
################
# SUPPORT VECTOR MACHINE FUNCTIONS
##################################
    
# perform classification for a set of train features/classes, a single test
# feature vector, and a particular classifier type
# PCA example adapted from:
# http://stackoverflow.com/questions/32194967/how-to-do-pca-and-svm-for-classification-in-python
def classify(train_feature_matrix, train_classes, test_features, classifier):
    if classifier == KNeighborsClassifier:
        model = classifier(23, weights='distance', p=1)
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == SVC:
        model = classifier(kernel='precomputed')
        model.fit(gram(train_feature_matrix, train_feature_matrix), train_classes.ravel())
        prediction = model.predict(gram(train_feature_matrix, test_features))[0]
    elif classifier == GaussianNB:
        model = classifier()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == SGDClassifier:
        model = classifier(loss='modified_huber', class_weight='balanced', penalty='l1')
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == RandomForestClassifier:
        model = classifier()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
        
    return genre_from_int(prediction)
    
# perform a 'leave one out' test on a particular classifier type using our data set
def leave_one_out(feature_list, glob, classifier, title):
    all_features = glob.get_features(feature_list)
    all_classes = glob.get_feature('class')
    
    class_pred, class_real = [], []
    
    vis.print_stars(newline=True)
    print("Testing " + title + " Classification with features:")
    print(feature_list)
    vis.print_dashes()
    sys.stdout.write("\r0 / %d samples processed (...)" % len(all_features))
    
    pca = PCA(whiten=True)
    all_features = pca.fit_transform(all_features)
    
    start = time.clock()
    
    for idx in range(len(all_features)):
        train_features = np.delete(all_features, idx, 0)
        train_classes = np.delete(all_classes, idx, 0)
        
        test_feature = np.transpose(all_features[idx,:]).reshape((1, train_features.shape[1]))
        test_class = np.transpose(all_classes[idx,:])
        
        predicted_class = classify(train_features, train_classes, test_feature, classifier)
        
        class_pred.append(predicted_class)
        class_real.append(genre_from_int(test_class))
    
        t = time.clock() - start
        time_per_iteration = t / (idx + 1)
        remaining = time_per_iteration * (len(all_features) - (idx + 1))
        
        sys.stdout.write("\r%d / %d samples processed (%02d:%02d:%02d left)" % 
            ((idx + 1), len(all_features), remaining / 3600, (remaining / 60) % 60, remaining % 60))
    
    return [class_pred, class_real]

if __name__ == '__main__':
    # load all songs into a glob
    glob = SongGlob()

    # all_features = ['eng', 'chroma', 'keystrength', 'brightness', 'zerocross', 'mfc', 'roughness', 'inharmonic', 'tempo', 'key']
    # 
    # all_features = list(powerset(all_features))
    # random.shuffle(all_features)
    # 
    # idx = 0
    # results = {}
    # max_acc = 0
    # max_feat = []
    # start = time.clock()
    # 
    # for feature_set in all_features:
    #     if len(feature_set) == 0:
    #         continue
    #     
    #     p1, r1, = leave_one_out(feature_set, glob, KNeighborsClassifier, "KNN")
    #     acc = vis.present_results(p1, r1, "KNN", print_results=False, show_results=True)
    #     if acc > max_acc:
    #         max_acc = acc
    #         max_feat = feature_set
    #         print()
    #         print(max_acc, max_feat)
    #         
    #     results[feature_set] = acc
    #     
    #     idx += 1
    #     
    #     t = time.clock() - start
    #     remaining = t * (len(all_features) / idx) - t
    #     
    #     sys.stdout.write("\r%d / %d permutations (%d:%02d left)" % (idx, len(all_features), remaining // 60, remaining % 60))
    # 
    # print("")
    # 
    # for set in sorted(results, key=results.get, reverse=True):
    #     print(set, results[set])
    # 
    
    params = ['eng', 'chroma', 'keystrength', 'zerocross', 'tempo', 'mfc', 'brightness', 'roughness', 'inharmonic']
    
    p1, r1 = leave_one_out(params, glob, KNeighborsClassifier, "K Nearest Neighbors")
    vis.present_results(p1, r1, "K Nearest Neighbors", print_results=True, show_results=False)
    
    p2, r2 = leave_one_out(params, glob, GaussianNB, "Gaussian Naive Bayes")
    vis.present_results(p2, r2, "Gaussian Naive Bayes", print_results=True, show_results=False)

    p3, r3 = leave_one_out(params, glob, SGDClassifier, "Stochastic Gradient Descent")
    vis.present_results(p3, r3, "Stochastic Gradient Descent", print_results=True, show_results=False)
    
    p4, r4 = leave_one_out(params, glob, RandomForestClassifier, "Random Forest")
    vis.present_results(p4, r4, "Random Forest", print_results=True, show_results=False)
    
    p5, r5 = leave_one_out(params, glob, SVC, "Support Vector Machine")
    vis.present_results(p5, r5, "Support Vector Machine", print_results=True, show_results=False)
