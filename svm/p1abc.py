from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import csv

data = []
x = []
y = []
val = []

# Boolean variables for running different parts of the code
p1a = False
p1b = False
p1c = False

def plotpoint(point, label = None):
	if label == 0:
		plt.plot(point[0], point[1], 'ro')
	elif label == 1:
		plt.plot(point[0], point[1], 'bo')
	else:
		plt.plot(point[0], point[1], 'g')

def plotpointtest(point, label = None):
	if label == 0:
		plt.plot(point[0], point[1], 'r+')
	elif label == 1:
		plt.plot(point[0], point[1], 'b+')
	else:
		plt.plot(point[0], point[1], 'g')

with open('data_a.csv',newline = '') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
	for row in reader:
		x.append(float(row[0]))
		y.append(float(row[1]))
		val.append(float(row[2]))

# plot the training data
if p1a:
	for i in range(len(x)):
		if val[i] == 0:
			plt.plot(x[i], y[i], 'ro')
		else:
			plt.plot(x[i], y[i], 'bo')

# train a linear SVM classifer and use it to classify and plot the test data
if p1b:
	for i in range(len(x)):
		data.append([x[i], y[i]])

	clf = svm.LinearSVC() # creates an instance of the Linear SVC class
	clf.fit(data, val) 	  # fits the data and their labels (val) using our SVM
					   	  # with a linear kernel

	testpoint1 = [2, 2]
	testpoint2 = [-1, 3]
	label1 = clf.predict([testpoint1])
	label2 = clf.predict([testpoint2])
	plotpointtest(testpoint1, label1)
	plotpointtest(testpoint2, label2)

# compute and plot the decision function
if p1c:
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# create grid to evaluate model
	xx = np.linspace(xlim[0], xlim[1], 30)
	yy = np.linspace(ylim[0], ylim[1], 30)
	YY, XX = np.meshgrid(yy, xx)
	xy = np.vstack([XX.ravel(), YY.ravel()]).T
	Z = clf.decision_function(xy).reshape(XX.shape)

	# plot decision boundary and margins
	ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
	           linestyles=['-'])
	# plot support vectors
	#ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
	#           linewidth=1, facecolors='none', edgecolors='k')

plt.show()

