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

class DataRun():
	def __init__(self):
		self.x = []
		self.y = []		

theta = np.linspace(0,2*np.pi)
run1 = DataRun()
run2 = DataRun()
run3 = DataRun()
run4 = DataRun()
run5 = DataRun()

run6 = DataRun()
run7 = DataRun()
run8 = DataRun()
run9 = DataRun()
run10 = DataRun()

for i in theta:
	circ_func(run1,i)
	circ_func(run2,i)
	circ_func(run3,i)
	circ_func(run4,i)
	circ_func(run5,i)

	circ_func2(run6,i)
	circ_func2(run7,i)
	circ_func2(run8,i)
	circ_func2(run9,i)
	circ_func2(run10,i)

plt.plot(run1.x,run1.y, 'ro')
plt.plot(run2.x,run2.y, 'ro')
plt.plot(run3.x,run3.y, 'ro')
plt.plot(run4.x,run4.y, 'ro')
plt.plot(run5.x,run5.y, 'ro')

plt.plot(run6.x,run6.y, 'bo')
plt.plot(run7.x,run7.y, 'bo')
plt.plot(run8.x,run8.y, 'bo')
plt.plot(run9.x,run9.y, 'bo')
plt.plot(run10.x,run10.y, 'bo')
plt.show()

runs = []
runs.append(run1)
runs.append(run2)
runs.append(run3)
runs.append(run4)
runs.append(run5)
runs.append(run6)
runs.append(run7)
runs.append(run8)
runs.append(run9)
runs.append(run10)

with open('data_b.csv','a+',newline ='') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(runs)):
		for j in range(len(runs[i].x)):
			csvwriter.writerow([runs[i].x[j], runs[i].y[j], 0])