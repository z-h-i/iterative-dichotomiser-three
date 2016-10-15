import sys
import pandas as pd
from trees.node import Node			# Node class for each node of the tree
from f_growtree import grow_tree	# the ID3 algorithm, more or less

'''
main(): completes parts a, b, and c of the assignment;
	the program takes two arguments: the training and
	test datasets
	arguments must either:
		1) be file names within the same directory
		as id3_zhiyue_wang.py, or
		2) be full directory paths
'''
def main(argv):
	if len(argv) is not 2:
		print("Please enter two file names")
		sys.exit()
	train_file = argv[0]
	test_file = argv[1]
	try:
		train = pd.read_table(train_file)
		test = pd.read_table(test_file)
	except:
		print("Not valid file names or files aren't in the current directory or not '.dat' files")
		sys.exit()
	
	tree = grow_tree(train)
	tree.print_tree()
	accuracy = tree.classify(train)
	print("\n")
	print("Accuracy on training set (" + str(len(train)) + 
		" instances):  " + str(round(accuracy * 100, 1)) + "%")
	accuracy = tree.classify(test)
	print("\n")
	print("Accuracy on test set (" + str(len(test)) + 
	" instances):  " + str(round(accuracy * 100, 1)) + "%")
		

if __name__ == "__main__":
	main(sys.argv[1:])
