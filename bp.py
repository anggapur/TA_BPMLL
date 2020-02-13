# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from cek_konvergen import  cekKonvergen
from copy import deepcopy
import os
import re
from validate import validation

import time

def enumarate_file(directory):
    return [filename for filename in sorted(os.listdir(directory))]

def transposing(data):
	return list(map(list, zip(*data)))

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	# print(dataset[0])
	# print(len(dataset))
	# for index,data in enumerate(dataset):
	# 	print(index+1)
	# 	print(len(data))
	# 	print(data)
	# print(dataset)
	# print('------')

	predict_keep = []
	actual_keep = []
	learning_time_keep = [None] * 10

	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for ix,fold in enumerate(folds):
		start = time.process_time()
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		predict_keep.append(predicted)
		actual_keep.append(actual)
		# accuracy = accuracy_metric(actual, predicted)
		# print("Predicted")
		# print(len(predicted))
		# print(predicted)
		# scores.append(accuracy)
		end = time.process_time()
		learning_time = str(end - start)
		learning_time_keep[ix] = float(learning_time)

	# return scores
	return {"predicted" : predict_keep , "actual" : actual_keep , "learning_time_keep" : learning_time_keep}

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    old = False
    i = 1
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            # print(network)
        # print(i, ' -----')
        # print(network)
        i+=1
        if(old is False):
            old = deepcopy(network)
            # print('old',old)
        else:
            new = deepcopy(network)
            # print('new',new)
            # print('old',old)
            hasil = (cekKonvergen(old,new,0.045))
            # print(hasil)
            old = deepcopy(network)
            if(hasil is True):
                break;



    # print(network)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # print(network)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    # print(network)
    return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    # print(network)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()

    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)

# Test Backprop on Seeds dataset
# seed(1)
# load and prepare data

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_filenames(directory):
	document_filenames = []
	for root, dirs, files in os.walk(directory):
		# print(files)
		files.sort(key=natural_keys)
		# print(files)
		for filename in files:
			# print('dataVSM/'+filename)
			document_filenames.append(directory+'/'+filename)
		return document_filenames

kelas = 10
# filename = "fitur_ekstraksi/data_bow/data_final_tf_100.txt"
folder_awal = "../data_2017-2019_10-label"
hl = 5
epoch = 50
learning_rate = 0.1
folder_hasil = "hasil_"+str(kelas)+"-label_"+str(hl)+"-hidden_"+str(epoch)+"-epoch_"+str(learning_rate)+"-lrate_"

#Make Folder
if not os.path.exists(folder_hasil):
    os.makedirs(folder_hasil)

for filename in enumarate_file(folder_awal):
	new_file = open(folder_hasil+"/hasil-"+filename, 'w')
	leng_feature = filename.split('_')
	leng_feature = int(leng_feature[3].strip('.txt'))
	print(leng_feature)

	hasild1 = [[],[],[],[],[],[],[],[],[],[]]
	actuald1 = [[],[],[],[],[],[],[],[],[],[]]
	learningtimed1 = []
	for iter in range(0,kelas):
		# print(iter)
		dataset = load_csv(folder_awal+"/"+filename)
		dataset = dataset[0:]

		new_dataset = []

		for d in dataset:
			n_d = d[0:-kelas]
			n_d_kelas = d[(leng_feature+iter)]
			n_d.append(n_d_kelas)
			new_dataset.append(n_d)

		dataset = new_dataset

		for i in range(len(dataset[0])-1):
			str_column_to_float(dataset, i)

		# convert class column to integers
		str_column_to_int(dataset, len(dataset[0])-1)
		# normalize input variables
		minmax = dataset_minmax(dataset)
		normalize_dataset(dataset, minmax)
		# evaluate algorithm
		n_folds = 10
		l_rate = learning_rate
		n_epoch = epoch
		n_hidden = hl
		# print('HIDDEN LAYER  : ' + str(hl))
		scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
		# print(scores['learning_time_keep'])
		# print('Scores: %s' % scores)
		# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

		for ix,h in enumerate(scores['predicted']):
			hasild1[ix].append(h)

		for ix,h in enumerate(scores['actual']):
			actuald1[ix].append(h)

		learningtimed1.append(scores['learning_time_keep'])

	# print(hasild1)
	# print(actuald1)
	learning_time_transpose = transposing(learningtimed1)
	# print(learning_time_transpose)

	final_learning_time = [sum(d) for d in learning_time_transpose]
	# print(final_learning_time)

	# print("%%%%%%%%%")
	for iz, data_z in enumerate(hasild1):
		# print(hasild1[iz])
		# print(transposing(hasild1[iz]))
		# print(actuald1[iz])
		# print(transposing(actuald1[iz]))
		validation_result = validation(transposing(hasild1[iz]), transposing(actuald1[iz]))
		print(str(validation_result) + "|" + str(final_learning_time[iz]))
		new_file.write(str(validation_result) + "|" + str(final_learning_time[iz]) + "\n")
		# print("-----")

	new_file.close()
