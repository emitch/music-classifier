import os, random, sys, time, copy
import numpy as np
import scipy.io as sio
from itertools import chain, combinations
import scipy
import math

# sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

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
            gram[i, j] = fast_cos_kernel(test_features[i], train_features[j])
    
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
    
# fast cosine kernel from 
# http://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
def fast_cos_kernel(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
    
# compute the cosine of the feature vectors
def cos_kernel(x1, x2):
    return -scipy.spatial.distance.cosine(x1, x2)
    
################
# SUPPORT VECTOR MACHINE FUNCTIONS
##################################
    
# perform classification for a set of train features/classes, a single test
# feature vector, and a particular classifier type
# PCA example adapted from:
# http://stackoverflow.com/questions/32194967/how-to-do-pca-and-svm-for-classification-in-python
def classify(train_feature_matrix, train_classes, test_features, classifier):
    if classifier == 1:
        model = KNeighborsClassifier(23, weights='distance', p=1)
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 2:
        model = VotingClassifier(estimators=[
            ('knn1', KNeighborsClassifier(10, weights='distance', p=1)),
            ('knn3', KNeighborsClassifier(30, weights='distance', p=1)),
            ('knn4', KNeighborsClassifier(50, weights='distance', p=1)),
            ('knn4', KNeighborsClassifier(70, weights='distance', p=1)),
            ('knn5', KNeighborsClassifier(90, weights='distance', p=1))],
            voting='soft')
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 3:
        model = GaussianNB()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 4:
        model = SGDClassifier(loss='modified_huber', class_weight='balanced', penalty='l1')
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 5:
        model = RandomForestClassifier()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 6:
        model = DecisionTreeClassifier()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 7:
        model = KNeighborsClassifier()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 8:
        model = SGDClassifier()
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 9:
        model = AdaBoostClassifier(n_estimators=100)
        model.fit(train_feature_matrix, train_classes.ravel())
        prediction = model.predict(test_features)[0]
    elif classifier == 10:
        model = SVC(kernel='precomputed')
        model.fit(gram(train_feature_matrix, train_feature_matrix), train_classes.ravel())
        prediction = model.predict(gram(train_feature_matrix, test_features))[0]
    
    return genre_from_int(prediction)
    
# perform a 'leave one out' test on a particular classifier type using our data set
def leave_one_out(feature_dict, glob, classifier, title):
    # feature_dict is a dictionary of feature names and a triple of booleans defining
    # which summary metrics to include respectively: (mean, std, measurewise)
    all_features = glob.get_features(feature_dict)
    all_classes = glob.get_feature('class', (True, True, True))
    
    class_pred, class_real = [], []
    
    vis.print_stars(newline=True)
    print("Testing " + title + " classification with features:")
    print(list(feature_dict.keys()))
    vis.print_dashes()
    sys.stdout.write("\r0 / %d samples processed (...)" % len(all_features))
    
    pca = LinearDiscriminantAnalysis()
    all_features = pca.fit_transform(all_features, all_classes.ravel())
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

def optimize(classifier, feature_dict):
    # full feature set
    all_features = ['eng', 'chroma', 'keystrength', 'brightness', 'zerocross', 
        'mfc', 'roughness', 'inharmonic', 'tempo', 'key']
    
    # get the power set of all_features
    all_features = list(powerset(all_features))
    random.shuffle(all_features)

    # initialize
    idx = 0
    results = {}
    max_acc = 0
    max_feat = []
    start = time.clock()
    
    # loop through each possible feature set
    for feature_set in all_features:
        if len(feature_set) == 0:
            continue

        p1, r1, = leave_one_out(feature_dict, glob, classifier, classifier.__name__)
        acc = vis.present_results(p1, r1, classifier.__name__, print_results=False, show_results=True)

        # print the running best combination
        if acc > max_acc:
            max_acc = acc
            max_feat = feature_set
            print()
            print(max_acc, max_feat)
            
        results[feature_set] = acc
        
        idx += 1
        
        t = time.clock() - start
        remaining = t * (len(all_features) / idx) - t
        
        sys.stdout.write("\r%d / %d permutations (%d:%02d left)" % (idx, len(all_features), remaining // 60, remaining % 60))
    
    print("")
    
    return sorted(results, key=results.get, reverse=True)

if __name__ == '__main__':
    # load all songs into a glob
    glob = SongGlob()
    
    # all params availible
    params = ['eng', 'chroma', 'keystrength', 'zerocross', 'tempo', 'mfc', 'brightness', 'roughness', 'inharmonic']

    param_dict = dict()
    for param in params:
        if param in ['inharmonic']:
            param_dict[param] = (True, True, False)
        elif param in ['chroma', 'keystrength']:
            param_dict[param] = (True, False, True)
        else:
            param_dict[param] = (True, True, True)
    
    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 1, "K Neighbors (Opt)")
    t = time.clock() - start
    vis.present_results(p, r, "K Neighbors (Opt)", t, print_results=True, show_results=True)
    
    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 2, "Voting")
    t = time.clock() - start
    vis.present_results(p, r, "Voting", t, print_results=True, show_results=True)
    
    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 3, GaussianNB.__name__)
    t = time.clock() - start
    vis.present_results(p, r, GaussianNB.__name__, t, print_results=True, show_results=True)
    
    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 4, "Stochastic Gradient Descent (Opt)")
    t = time.clock() - start
    vis.present_results(p, r, "Stochastic Gradient Descent (Opt)", t, print_results=True, show_results=True)
    
    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 5, RandomForestClassifier.__name__)
    t = time.clock() - start
    vis.present_results(p, r, RandomForestClassifier.__name__, t, print_results=True, show_results=True)

    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 6, "Decision Tree")
    t = time.clock() - start
    vis.present_results(p, r, "Decision Tree", t, print_results=True, show_results=True)

    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 7, "K Neighbors (Def)")
    t = time.clock() - start
    vis.present_results(p, r, "K Neighbors (Def)", t, print_results=True, show_results=True)

    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 8, "Stochastic Gradient Descent (Def)")
    t = time.clock() - start
    vis.present_results(p, r, "Stochastic Gradient Descent (Def)", t, print_results=True, show_results=True)

    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 9, "Ada Boost (n=100)")
    t = time.clock() - start
    vis.present_results(p, r, "Ada Boost (n=100)", t, print_results=True, show_results=True)

    start = time.clock()
    p, r = leave_one_out(param_dict, glob, 10, "Support Vector Machine")
    t = time.clock() - start
    vis.present_results(p, r, "Support Vector Machine", t, print_results=True, show_results=True)
