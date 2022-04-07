# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:50:44 2020

@author: Nan
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import tree
from sklearn.externals.six import StringIO
import graphviz
import pydotplus
# dummy data:
df = pd.DataFrame({'col1':[0,1,2,3],'col2':[3,4,5,6],'dv':[0,1,0,1]})

# create decision tree
DTC = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
DTC.fit(df.ix[:,:2], df.dv)
print(DTC.predict([[0,3]]))
print(DTC.classes_)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

tree_to_code(DTC, df.columns)

def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print(node)
              
dot_data = StringIO()              
dot_data = tree.export_graphviz(DTC, out_file=None, 
                         feature_names=['col1','col2'],  
                         class_names=['A','B'],  
                         filled=True, rounded=True,  
                         special_characters=True)  

with open("dtc.dot",'w+',)as f:
    f = tree.export_graphviz(DTC,out_file=f)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(" DecisionTree_test.pdf")
#pdf = graphviz.Source(dot_data)  
#pdf.view()
 

