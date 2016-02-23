from collections import defaultdict
import random, sys, ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.interpolate import UnivariateSpline

num_plots = 0
categories = ['blues', 'classical', 'country', 'disco', 'hiphop', \
    'jazz', 'metal', 'pop', 'reggae', 'rock']
    
def print_stars(newline=False):
    if newline:
        print("")
    
    print("****************")
    
def print_dashes(newline=False):
    if newline:
        print("")
    
    print("--------")

def genre_as_int(genre):
    return categories.index(genre)

def hist_of_points(values, feature_name, graph_title):
    global num_plots
    num_plots += 1
    
    plt.figure(num_plots)
    plt.hist(values, facecolor='dimgray')
    
    plt.xlabel(feature_name)
    plt.ylabel('Occurrences')
    plt.title(title)
    
    plt.show()
    
    """
    Must implement *load_data()* and *get_feature()* for this function
    """
#def vis_feature(feature_name, categorization_feature):
#    all_data = # load_data()
#    feature_set = {}
#    for struct in all_data:
#        feature = get_feature(struct, feature_name)
        # if get_feature(struct, feature_name)
        
# http://matplotlib.org/examples/color/colormaps_reference.html
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def present_results(pred_data, ref_data, title, print_results=False, show_results=False):
    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)
    correct_percents = defaultdict(float)
    
    correct_total = 0

    # get the categories in the reference list
    for pred, ref in zip(pred_data, ref_data):  
        if pred == ref:
            correct_total += 1
            correct_counts[ref] += 1
            incorrect_counts[ref] += 0
        else:
            correct_counts[ref] += 0
            incorrect_counts[ref] += 1
    
    percent_correct = (correct_total / len(ref_data)) * 100
    
    if print_results:
        print_dashes(newline=True)
        print(title + ":")
        print("Overall accuracy: {:.2f} % correct".format(percent_correct))
        print_stars()
    
    if not show_results:
        return percent_correct
    
    for key in correct_counts:
        correct_percents[key] = correct_counts[key] / (correct_counts[key] + incorrect_counts[key])

    # reference list of the categories we've seen
    categories = list(correct_counts.keys())
    categories.sort()
    
    # x and y values for charting correct counts
    correct_x = []
    correct_y = []
    
    for cat in categories:
        correct_x.append(cat)
        correct_y.append(correct_percents[cat])
    
    # now we need to display the data
    # show the correct counts for each category
    
    N = 10
    ind = np.arange(1, N + 1)    # the x locations for the groups
    width = 0.5       # the width of the bars: can also be len(x) sequence
    colors = ['dimgray', 'red', 'brown', 'darkgray', 'yellow', 'palegreen', 
        'seagreen', 'deepskyblue', 'navy', 'deeppink']
    
    global num_plots
    num_plots += 1
    plt.figure(num_plots)
    p_correct = plt.bar(ind, correct_y, width, color='.75', edgecolor='k')

    plt.ylabel('% Correct')
    plt.title(title)
    plt.xticks(ind + width / 2., correct_x, rotation='40')
    plt.yticks(np.arange(0, 1.05, .1))
    
    for idx in range(len(pred_data)):
        pred_data[idx] = genre_as_int(pred_data[idx])
        ref_data[idx] = genre_as_int(ref_data[idx])
    
    # Compute confusion matrix
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm = confusion_matrix(ref_data, pred_data)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, title=title)

    plt.show()
    
    # show the misses organized by their actual category
    # num_plots += 1
    # plt.figure(num_plots)
    # 
    # plots = []
    # for index, cat in enumerate(categories):
    #     plots.append(plt.bar(ind, pred_dict[cat], width, color=colors[index], bottom = pred_cumul)[0])
    #     for idx, val in enumerate(pred_dict[cat]):
    #         pred_cumul[idx] += val
    # 
    # plt.ylabel('# Incorrect Predictions')
    # plt.xlabel('Actual Genre')
    # plt.title('Incorrect Predictions by Actual Genre')
    # plt.xticks(ind + width / 2., categories, rotation='40')
    # plt.yticks(np.arange(0, int(max(pred_cumul) * 1.5), 10))
    # plt.legend(plots, categories, ncol=4)
    # 
    # # show the misses organized by their predicted category
    # 
    # num_plots += 1
    # plt.figure(num_plots)
    # 
    # plots = []
    # for index, cat in enumerate(categories):
    #     plots.append(plt.bar(ind, ref_dict[cat], width, color=colors[index], bottom = ref_cumul)[0])
    #     for idx, val in enumerate(ref_dict[cat]):
    #         ref_cumul[idx] += val
    # 
    # plt.ylabel('# Incorrect Predictions')
    # plt.xlabel('Predicted Genre')
    # plt.title('Incorrect Predictions by Predicted Genre')
    # plt.xticks(ind + width / 2., categories, rotation='40')
    # plt.yticks(np.arange(0, int(max(ref_cumul) * 1.4), 10))
    # plt.legend(plots, categories, ncol=4)
    # 
    # plt.show()
    # 
    # return percent_correct

# def plot_fv(fv, color=):
#     """ plot a feature vector in 2D space as simply a curve with feature
#     on the x-axis and value on the y-axis """
#     plt.
# 
#     return handle

if __name__ == '__main__':
    present_results(sys.argv[1], sys.argv[2])
