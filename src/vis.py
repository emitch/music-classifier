from collections import defaultdict
import random
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

def present_results(pred_path, ref_path):
    # get the individual entries from each file
    pred_data = [line.rstrip('\n') for line in open(pred_path)]
    ref_data = [line.rstrip('\n') for line in open(ref_path)]

    correct_counts = defaultdict(int)

    # get the categories in the reference list
    for pred, ref in zip(pred_data, ref_data):
        if len(ref) == 0:
            continue
        if len(pred) == 0:
            continue
        
        if pred == ref:
            correct_counts[ref] += 1
            
    # reference list of the categories we've seen
    categories = list(correct_counts.keys())
    categories.sort()
    
    # x and y values for charting correct counts
    correct_x = []
    correct_y = []

    for cat in categories:
        correct_x.append(cat)
        correct_y.append(correct_counts[cat])
    
    ref_dict = {}    # arranged by real category
    pred_dict = {}   # arranged by predicted category

    # get incorrect prediction data
    for pred, ref in zip(pred_data, ref_data):
        if pred != ref:
            ref_idx = categories.index(ref)
            pred_idx = categories.index(pred)
        
            if ref not in ref_dict:
                ref_dict[ref] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if pred not in pred_dict:
                pred_dict[pred] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
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

    plt.ylabel('Number Correct')
    plt.title('Correct Predictions by Category')
    plt.xticks(ind + width / 2., correct_x)
    plt.yticks(np.arange(0, 100, 10))
    
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
    plt.xticks(ind + width / 2., categories)
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
    plt.xticks(ind + width / 2., categories)
    plt.yticks(np.arange(0, 65, 10))
    plt.legend(plots, categories, ncol=4)
    
    plt.show()

if __name__ == '__main__':
    present_results(sys.argv[1], sys.argv[2])
