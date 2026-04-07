from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import csv

def outputfunc(input):
	#Random circle around -1,-1
	rand_rad = np.random.rand()
	return 0.5*np.cos(input), 1.5*np.sin(input)

def circ_func(run,theta):
	x,y = outputfunc(theta)
	#Random circle around 1,1
	run.x.append(x+1.1)
	run.y.append(y-1.1)

def circ_func2(run,theta):
	x,y = outputfunc(theta)
	#Random circle around 1,1
	if x < 0:
		x+=1
	else:
		x+= -1
	run.x.append(x)
	run.y.append(y+0.5)

class DataRun():
	def __init__(self):
		self.x = []
		self.y = []		

iteration = np.linspace(0,2*np.pi)
run1 = DataRun()

for i in iteration:
	circ_func2(run1,i)

plt.plot(run1.x,run1.y, 'ro')
plt.show()

runs = []
runs.append(run1)

with open('data_test_1def.csv','w',newline ='') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(runs)):
		for j in range(len(runs[i].x)):
			csvwriter.writerow([runs[i].x[j], runs[i].y[j]])