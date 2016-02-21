from collections import defaultdict
import random
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

num_plots = 0

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

def present_results(pred_data, ref_data, title, print_only=False):
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
            
    print("\n*********************************")
    print(title)
    print("Overall accuracy: {:.2f} % correct".format(correct_total / len(ref_data) * 100))
    print("*********************************")
    
    if print_only:
        return

    for key in correct_counts:
        correct_percents[key] = correct_counts[key] / (correct_counts[key] + incorrect_counts[key])

    # reference list of the categories we've seen
    categories = list(correct_counts.keys())
    categories.sort()
    
    # x and y values for charting correct counts
    correct_x = []
    correct_y = []
    
    ref_dict = {}    # arranged by real category
    pred_dict = {}   # arranged by predicted category
    ref_cumul = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred_cumul = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for cat in categories:
        ref_dict[cat] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred_dict[cat] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        correct_x.append(cat)
        correct_y.append(correct_percents[cat])

    # get incorrect prediction data
    for pred, ref in zip(pred_data, ref_data):
        if pred != ref:
            ref_idx = categories.index(ref)
            pred_idx = categories.index(pred)
            
            ref_dict[ref][pred_idx] += 1
            pred_dict[pred][ref_idx] += 1
    
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

    plt.ylabel('%% Correct')
    plt.title('Correct Predictions by Category')
    plt.xticks(ind + width / 2., correct_x, rotation='40')
    plt.yticks(np.arange(0, 1.05, .1))
    
    # show the misses organized by their actual category
    num_plots += 1
    plt.figure(num_plots)
    
    plots = []
    for index, cat in enumerate(categories):
        plots.append(plt.bar(ind, pred_dict[cat], width, color=colors[index], bottom = pred_cumul)[0])
        for idx, val in enumerate(pred_dict[cat]):
            pred_cumul[idx] += val
    
    plt.ylabel('# Incorrect Predictions')
    plt.xlabel('Actual Genre')
    plt.title('Incorrect Predictions by Actual Genre')
    plt.xticks(ind + width / 2., categories, rotation='40')
    plt.yticks(np.arange(0, int(max(pred_cumul) * 1.5), 10))
    plt.legend(plots, categories, ncol=4)
    
    # show the misses organized by their predicted category
    
    num_plots += 1
    plt.figure(num_plots)
    
    plots = []
    for index, cat in enumerate(categories):
        plots.append(plt.bar(ind, ref_dict[cat], width, color=colors[index], bottom = ref_cumul)[0])
        for idx, val in enumerate(ref_dict[cat]):
            ref_cumul[idx] += val

    plt.ylabel('# Incorrect Predictions')
    plt.xlabel('Predicted Genre')
    plt.title('Incorrect Predictions by Predicted Genre')
    plt.xticks(ind + width / 2., categories, rotation='40')
    plt.yticks(np.arange(0, int(max(ref_cumul) * 1.4), 10))
    plt.legend(plots, categories, ncol=4)
    
    plt.show()

if __name__ == '__main__':
    present_results(sys.argv[1], sys.argv[2])
