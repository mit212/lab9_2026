from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import csv

def outputfunc(input):
	#Random circle around -1,-1
	rand_rad = np.random.rand()
	return rand_rad*np.cos(input), rand_rad*np.sin(input)

def circ_func(run,theta):
	x,y = outputfunc(theta)
	#Random circle around 1,1
	run.x.append(x+1.1)
	run.y.append(y-1.1)

def circ_func2(run,theta):
	x,y = outputfunc(theta)
	#Random circle around 1,1
	run.x.append(x-1.1)
	run.y.append(y+1.1)

def square_func1(run,i):
	if i < 13:
		run.x.append(0.1+i/13)
		run.y.append(0.1)
	elif i < 26: 
		run.x.append(0.1+1)
		run.y.append(0.1+(i-12)/13)
	elif i <38:
		run.x.append(1.2-(i-24)/13)
		run.y.append(1.2)
	else:
		run.x.append(0.1)
		run.y.append(1.25-(i-37)/13)
class DataRun():
	def __init__(self):
		self.x = []
		self.y = []		

iteration = np.linspace(0,50)
run1 = DataRun()

for i in iteration:
	square_func1(run1,i)

plt.plot(run1.x,run1.y, 'ro')
plt.show()

runs = []
runs.append(run1)

with open('data_test_1abc.csv','w',newline ='') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(runs)):
		for j in range(len(runs[i].x)):
			csvwriter.writerow([runs[i].x[j], runs[i].y[j]])