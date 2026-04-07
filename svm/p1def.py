from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import csv

data = []
x = []
y = []
val = []
p1d = True
p1e = True
p1f = True

def plotpoint(point, label = None):
	if label == 0:
		plt.plot(point[0],point[1],'ro')
	elif label == 1:
		plt.plot(point[0],point[1],'bo')
	else:
		plt.plot(point[0],point[1],'g')
def plottestpoint(point, label = None):
	if label == 0:
		plt.plot(point[0],point[1],'r+')
	elif label == 1:
		plt.plot(point[0],point[1],'b+')
	else:
		plt.plot(point[0],point[1],'g')
with open('data_b.csv',newline = '') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
	for row in reader:
		x.append(float(row[0]))
		y.append(float(row[1]))
		val.append(float(row[2]))

if p1d:
	for i in range(len(x)):
		if val[i] == 0:
			plt.plot(x[i],y[i],'ro')
		else:
			plt.plot(x[i],y[i],'bo')

if p1e:
	for i in range(len(x)):
		data.append([x[i], y[i]])

	clf = svm.SVC()

	# Question 2
	# TODO: Try changing the values of C and gamma!
	# clf = svm.SVC(C = 1.0, kernel = 'rbf', gamma = 'scale')
	
	# Question 3
	# TODO: Try changing the kernel!
	# clf = svm.SVC(kernel = 'poly')
	
	# Question 4
	# TODO: Try changing the polynomial kernel degree!
	# clf = svm.SVC(kernel = 'poly', degree = 4)

	clf.fit(data,val)

	testpoint1 = [0,0]
	label1 = clf.predict([testpoint1])
	plottestpoint(testpoint1,label1)

if p1f:
	# plot the decision function
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
	ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
	           linestyles=['--','-','--'])
	# plot support vectors
	ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
	           linewidth=1, facecolors='none', edgecolors='g')

plt.show()


