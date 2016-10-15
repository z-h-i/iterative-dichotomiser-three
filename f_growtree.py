import pandas as pd
from functools import reduce
from itertools import chain as ch
from trees.node import Node

'''
grow_tree(): an ID3 implementation for binary classification;
	given a dataset with binary attribute and target labels,
	grow_tree recursively divides the dataset via the
	minimal conditional entropy (i.e., maximum gain) heuristic.
	Each subdivision creates a node holding attribute name and 
	data subset information until division is no longer possible,
	at which case the node is a leaf node.
	RETURN: a binary decision tree
'''

def grow_tree(data, glob_freq_label = None, name = "root", branch = None):
	
	'''
	data: a pandas DataFrame with a subset of the data; 
		  the data is corralled by an attribute with minimal 
		  conditional entropy in the previous iteration of grow_tree();
		  1st iteration: all data
		  
	glob_freq_label: the most frequent target class;
		  
	name: the attribute criterion the node branched from; 
		  each name is the attribute with minimal conditional entropy
		  in the previous iteration;
		  1st iteration: "root"
		  
	branch: the attribute criterion's label;
		  T (1) or F (0), from the last iteration's 
		  attribute with minimal conditional entropy;
		  1st iteration: None
	'''
	
	node = Node(data)				# node with data subset
	node.name = name				# name it via its branching attribute & label
	node.branch_case = str(branch) if branch is not None else None
	rows, cols = node.data.shape	# grab the dimensions of the ata
	H = node.entropy()				# compute target label entropy
	
	if name is "root":						# calculate most common label for entire dataset
		glob_freq_label = 1 if float(sum(data.iloc[:, cols - 1])) / len(data) >= 0.5 else 0	
	loc_freq_label = (1 if float(sum(node.data.iloc[:, cols - 1])) 
		/ len(node.data) >= .5 else 0)		# calculate most common label within subset of data
	
	if H == 0:														# if there is 0 uncertainty, then
		node.label = 1 if node.data.iloc[0, cols - 1] == 1 else 0 	# the current node is a leaf, and
		return node												  ### its label is the target label
		
	if cols == 1:													# if there are no more attributes, then the current
		node.label = glob_freq_label if H == 1 else loc_freq_label	# node's label is the most common label in the data subset,
		return node												  ### unless impurity is maximum, then it is the most common
																	# label in the entire data
																	
	identical_attr = (map(lambda x, y: x == y, node.data.iloc[0, :(cols-1)], node.data.iloc[i, :(cols-1)]) for i in range(rows))
	identical_attr = list(ch.from_iterable(identical_attr))			# compare each row with the first via map and de-nest
	identical_attr = reduce(lambda x, y: x and y, identical_attr)	# the nested T and F values via from_iterable so that
																	# reduce can determine if all rows are identical
	if identical_attr:
		node.label = glob_freq_label if H == 1 else loc_freq_label	# if all examples' attributes are identical,
		return node												  ### return the node and set its label depending on H
		
	best_attr, H_cond = node.min_conditional_entropy()				# compute the value and attribute of minimal conditional H
																	
	data_true = pd.DataFrame(node.data[node.data.iloc[:, best_attr] == 1])				# split data into two new datasets via true and
	data_true = pd.DataFrame(data_true.drop(data_true.columns[best_attr], axis = 1))	# false labels of the best attribute, then
	data_false = pd.DataFrame(node.data[node.data.iloc[:, best_attr] == 0])				# drop the attribute from the datasets, preventing
	data_false = pd.DataFrame(data_false.drop(data_false.columns[best_attr], axis = 1))	# the tree from reusing redundant attributes
	
	name_true = node.data.columns[best_attr]					# set the branching names for the true 
	name_false = node.data.columns[best_attr]					# and false branches of the next iteration	
	
	node.branch_1 = grow_tree(data_true, glob_freq_label, name_true, 1)					# and create the branches via grow_tree
	node.branch_0 = grow_tree(data_false, glob_freq_label, name_false, 0)
	
	return node
	

	
