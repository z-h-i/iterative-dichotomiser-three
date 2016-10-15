import numpy as np
import scipy.stats as sci
import pandas as pd
from itertools import chain as ch

'''
Node: represents the node of an ID3 decision tree for binary classification;
	entropy(): calculates log(base2) Shannon's H for a dataset
	min_conditional_entropy(): calculates the minimal conditional
		H for all attributes given a dataset
	classify(): tests trained decision tree accuracy against a
		dataset
	print_tree(): prints a diagram of the tree
'''
class Node(object):

	data = None			# subset of the dataset the node holds
	name = None			# name of the attribute defining the node
	label = None		# class label if node is a leaf: 0 (F) or 1 (T)
	branch_0 = None		# child node for 0 (F) case of best_attr
	branch_1 = None		# child node for 1 (T) case of best_attr
	branch_case = None	# label of node's attribute: 0 (F) or 1 (T)
	correct = 0			# number of correct classifications
	
	def __init__(self, dataframe = None):							# each node stores a subset of the data
		self.data = dataframe
		
	'''
	entropy(): calculates, well, the entropy;
		particularly: log base 2 entropy - via scipy.stats.entropy() - with
		respect to the data's target label
		RETURN: entropy
	'''
	def entropy(self, data = None):
		if data is None:
			data = self.data
		num_obs, num_col = data.shape
		p = float(sum(data.iloc[:, num_col - 1])) / num_obs			# calculate frequency of the target attribute == True
		p = [p, 1 - p]												# and == False
		return sci.entropy(p, base = 2)							  ### calculate log base-2 entropy
		
	'''
	min_conditional_entropy(): calculates the minimal conditional entropy
		given the node's data;
		it does so the dumb way of checking the entropy and frequency of
		each label for each attribute, taking some dot products, then
		letting numpy find the minimal value via min()
		RETURN: (index of attribute with minimal conditional entropy, 
				minimal conditional entropy)
	'''
	def min_conditional_entropy(self):
		data = self.data
		num_attr = len(data.iloc[0, :]) - 1											# check the number of attributes
		sums_attr = [sum(x) for x in (data.iloc[:, i] for i in range(num_attr))]	# grab attribute == True frequencies,
		probs_attr = [float(x) / len(data) for x in sums_attr]						# for all attributes
		pq_attr = [probs_attr, [1 - p for p in probs_attr]]							# calculate == False frequencies
		pq_attr = np.matrix(pq_attr)
		attr_branch_entr = []
		cond_entr = []
		
		for i in range(num_attr):									# for each attribute:
			if pq_attr[0, i] == 1:										# if it's True for all data,
				attr_branch_entr.append([self.entropy(data), 0])		# its entropy for True is the target label's entropy
			elif pq_attr[1, i] == 1:									# if it's False for all data,
				attr_branch_entr.append([0, self.entropy(data)])		# its entropy for False is the target label's entropy
			else:
				attr_T = self.entropy(data[data.iloc[:, i] == 1])		# else, calculate the entropy with respect to True and
				attr_F = self.entropy(data[data.iloc[:, i] == 0])		# False labels of the attribute
				attr_branch_entr.append([attr_T, attr_F])
				
		attr_branch_entr = np.matrix(attr_branch_entr)
		for i in range(num_attr):													# for each attribute:
			cond_entr.append(np.dot(attr_branch_entr[i, :], pq_attr[:, i]))				# calculate a dot product of frequencies
		cond_entr = [matrix.tolist() for matrix in cond_entr]							# and entropies of each label
		
		cond_entr = np.array(list(ch.from_iterable(ch.from_iterable(cond_entr))))	# put everything into a cleaner container
		best_attr = list(ch.from_iterable(np.where(cond_entr == cond_entr.min())))	# grab the minimal value's index
		random_best = np.random.randint(0, len(best_attr))							# grab a random one if there are multiple 
		return best_attr[random_best], cond_entr[best_attr[random_best]]		  ### return the minimal conditional entropy & index
		
	'''
	classify(): computes classification accuracy given a test dataset;
		The tree must be trained first, of course, via grow_tree(train)
		RETURN: accuracy 
	'''
	def classify(self, data):
		if self.label is not None:													# if there's a label for a node
			rows, cols = data.shape														# grab the test data dimensions
			true_classes = sum(data.iloc[:, cols - 1])									# count the True label cases to
			Node.correct += (rows - true_classes) if self.label == 0 else true_classes	# add to correct classification count
		else:																		# if not,
			self.branch_1.classify(pd.DataFrame(data[data[self.branch_1.name] == 1]))	# divide the data with the node's
			self.branch_0.classify(pd.DataFrame(data[data[self.branch_0.name] == 0]))	# branches' attributes, and classify those
			if self.name is "root":								# if it's the root node
				accuracy = float(Node.correct) / len(data)			# calculate accuracy
				Node.correct = 0									# reset the count for correct classifications
				return accuracy									  ### return the accuracy

	'''
	print_tree(): prints out a diagram of sorts of a decision tree;
		the tree must be trained first via grow_tree(train)
	'''
	def print_tree(self, depth = 0):		# depth keeps a count of depth for pretty printing
		if type(self.label) is type(0):		# prints leaf nodes
			print(depth * "| " + self.name + " = " + self.branch_case + " :  " + str(self.label))
		elif self.name is not "root":					# prints non-leaf nodes
			print(depth * "| " + self.name + " = " + self.branch_case + " :  ")
			self.branch_0.print_tree(depth + 1)			# increase depth
			self.branch_1.print_tree(depth + 1)
		else:
			self.branch_0.print_tree(depth)				# no depth increase for root node
			self.branch_1.print_tree(depth)				# due to formatting constraints
