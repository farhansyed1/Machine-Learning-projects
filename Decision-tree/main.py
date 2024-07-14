import monkdata as m
import matplotlib.pyplot as plt
import numpy as np
import dtree as d
from PyQt5 import QtWidgets
import drawtree_qt5 as qt5

""" print(d.entropy(m.monk1))
print(d.entropy(m.monk2))
print(d.entropy(m.monk3))
print("GAIJN")
print(d.averageGain(m.monk3, m.attributes[0]))
print(d.averageGain(m.monk3, m.attributes[1]))
print(d.averageGain(m.monk3, m.attributes[2]))
print(d.averageGain(m.monk3, m.attributes[3]))
print(d.averageGain(m.monk3, m.attributes[4]))
print(d.averageGain(m.monk3, m.attributes[5]))

print("Decision trees")
t1= d.buildTree(m.monk1, m.attributes)
t2= d.buildTree(m.monk2, m.attributes)
t3= d.buildTree(m.monk3, m.attributes)
print(1-d.check(t1, m.monk1test))
print(1-d.check(t1, m.monk1))
print(1-d.check(t2, m.monk2test))
print(1-d.check(t2, m.monk2))
print(1-d.check(t3, m.monk3test))
print(1-d.check(t3, m.monk3))
 """


import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]
    


def pruning(monk, monkTest):
    prunedTrees = []
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for frac in fractions:   
        monktrain, monkval = partition(monk, frac)
        tree = d.buildTree(monktrain, m.attributes)
        alternatives=d.allPruned(tree) 
        
        maximumPerformance = d.check(tree,monkval) # istället för =1
        bestTree = tree
        currPerformance = 0

        for alternative in alternatives: 
            currPerformance = d.check(alternative, monkval)
            if currPerformance > maximumPerformance: 
                maximumPerformance = currPerformance
                bestTree = alternative

        prunedTrees.append(1-d.check(bestTree, monkTest))

    return prunedTrees
        
            
monk1_pruned =  np.transpose([pruning(m.monk1, m.monk1test) for i in range(500)])
monk3_pruned =  np.transpose([pruning(m.monk3, m.monk3test) for i in range(500)])

# Calculate mean errors and standard deviations
mean_errors_monk1 = np.mean(monk1_pruned, axis=1)
std_devs_monk1 = np.std(monk1_pruned, axis=1)

mean_errors_monk3 = np.mean(monk3_pruned, axis=1)
std_devs_monk3 = np.std(monk3_pruned, axis=1)

# Plotting for monk1
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

plt.errorbar(fractions, mean_errors_monk1, yerr=std_devs_monk1, fmt='o', label='Monk1')
plt.plot(fractions, mean_errors_monk1, linestyle='-', marker='o', color='g')
plt.xlabel('Fractions')
plt.ylabel('Mean Error')
plt.title('Mean Error vs Fractions (Monk1)')
plt.legend()
plt.show()

# Plotting for monk3
plt.errorbar(fractions, mean_errors_monk3, yerr=std_devs_monk3, fmt='o', label='Monk3')
plt.plot(fractions, mean_errors_monk3, linestyle='-', marker='o', color='r')
plt.xlabel('Fractions')
plt.ylabel('Mean Error')
plt.title('Mean Error vs Fractions (Monk3)')
plt.legend()
plt.show()

# Plotting both
plt.errorbar(fractions, mean_errors_monk1, yerr=std_devs_monk1, fmt='o', label='Monk1')
plt.plot(fractions, mean_errors_monk1, linestyle='-', marker='o', color='b')

plt.errorbar(fractions, mean_errors_monk3, yerr=std_devs_monk3, fmt='o', label='Monk3')
plt.plot(fractions, mean_errors_monk3, linestyle='-', marker='o', color='r')

plt.xlabel('Fractions')
plt.ylabel('Mean Error')
plt.title('Mean Error vs Fractions')
plt.legend()
plt.show()