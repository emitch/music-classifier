from collections import defaultdict
import random
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

def present_results(pred_data, ref_data):
    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)
    correct_percents = defaultdict(float)

    # get the categories in the reference list
    for pred, ref in zip(pred_data, ref_data):  
        if pred == ref:
            correct_counts[ref] += 1
            incorrect_counts[ref] += 0
        else:
            correct_counts[ref] += 0
            incorrect_counts[ref] += 1

    for key in correct_counts:
        for index, value in enumerate(correct_counts[key]):
            correct_percents[key][index] = correct_counts[key] / (correct_counts[key] + incorrect_counts[key])

    print(correct_percents)

    # reference list of the categories we've seen
    categories = list(correct_counts.keys())
    categories.sort()
    
    # x and y values for charting correct counts
    correct_x = []
    correct_y = []
    
    ref_dict = {}    # arranged by real category
    pred_dict = {}   # arranged by predicted category

    print(categories)

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
    
    plt.figure(1)
    p_correct = plt.bar(ind, correct_y, width, color='.75', edgecolor='k')

    plt.ylabel('%% Correct')
    plt.title('Correct Predictions by Category')
    plt.xticks(ind + width / 2., correct_x, rotation='40')
    plt.yticks(np.arange(0, 1.05, .1))
    
    plt.show()
    
    # show the misses organized by their real category
    
    plt.figure(2)
    
    plots = []
    for index, cat in enumerate(categories):
        if index == 0:
            plots.append(plt.bar(ind, pred_dict[cat], width, color=colors[index])[0])
        else:
            plots.append(plt.bar(ind, pred_dict[cat], width, color=colors[index], bottom = pred_dict[categories[index - 1]])[0])

    plt.ylabel('# Incorrect Predictions')
    plt.xlabel('Actual Genre')
    plt.title('Incorrect Predictions by Actual Genre')
    plt.xticks(ind + width / 2., categories, rotation='40')
    plt.yticks(np.arange(0, 65, 10))
    plt.legend(plots, categories, ncol=4)
    
    plt.show()
    
    # show the misses organized by their predicted category
    
    plt.figure(3)
    
    plots = []
    for index, cat in enumerate(categories):
        if index == 0:
            plots.append(plt.bar(ind, ref_dict[cat], width, color=colors[index])[0])
        else:
            plots.append(plt.bar(ind, ref_dict[cat], width, color=colors[index], bottom = ref_dict[categories[index - 1]])[0])

    plt.ylabel('# Incorrect Predictions')
    plt.xlabel('Predicted Genre')
    plt.title('Incorrect Predictions by Predicted Genre')
    plt.xticks(ind + width / 2., categories, rotation='40')
    plt.yticks(np.arange(0, 65, 10))
    plt.legend(plots, categories, ncol=4)
    
    plt.show()

if __name__ == '__main__':
    present_results(sys.argv[1], sys.argv[2])
