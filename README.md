# Lab 9: Machine Learning

2.12/2.120 Intro to Robotics  
Spring 2025[^1]

<details>
  <summary>Table of Contents</summary>

- [1 Software Set Up](#1-software-set-up)
  - [1.1 Scikit-Learn](#11-scikit-learn)
  - [1.2 Tensorflow](#12-tensorflow)
- [2 Support Vector Machine (SVM)](#2-support-vector-machine-svm)
  - [2.1 Linearly Separable Case](#21-linearly-separable-case)
  - [2.2 Nonlinear SVM](#22-nonlinear-svm)
- [3 Neural Network (NN)](#3-neural-network-nn)
- [4 Convolutional Neural Network (CNN)](#4-convolutional-neural-network-cnn)
- [5 Submission to Canvas](#5-submission-to-canvas)

</details>

In this lab, you will experiment with machine learning techniques on your own. Please submit a PDF of the screenshots and answers to the "Lab 9: Machine Learning" assignment on Canvas **by Sunday, April 13, 11:59 p.m.** If you have any questions, feel free to reach out to the TAs or LAs via email (2.12-lab-staff-s25@mit.edu).

## 1 Software Set Up

If you need to specify a specific Python version to install a library to, and `pip` or `pip3` by itself isn't doing so, you can use `python -m pip install [library to install]` to install libraries, replacing `python` with `python3` if necessary.

### 1.1 Scikit-Learn

To install Scikit-Learn, enter `pip3 install scikit-learn` in your terminal.

### 1.2 Tensorflow

To install Tensorflow, enter `pip3 install tensorflow` in your terminal. If this worked for you, then ignore the information below.

**The section below goes through debugging steps to install TensorFlow. If the below doesn't work for you, or if you don't have time, use `classifier_pytorch.py` instead and install the pytorch libraries: `pip install torch` and `pip install torchvision`**.

If you have a Windows computer and you are running VS Code on your Windows machine, then you may have to perform the following extra steps. Alternatively, you can try running this lab through WSL2, in which case you won't have to do these steps.

Make sure you are using a Python version that is 3.7 - 3.10. Remeber that you can check your Python version by entering `python -V` or `python3 -V` in the terminal. Anything above Tensorflow 2.10 is not supported on the GPU on Windows Native, and Tensorflow 2.10 is only compatible with Python version 3.7-3.10. See more Tensorflow version compatibility information [here](https://www.tensorflow.org/install/source_windows#cpu). If you need to downgrade your Python version see this [link](https://medium.com/@codelancingg/how-to-downgrade-python-version-fb7b9087e776). 

Once you confirm your Python version is somewhere between 3.7 - 3.10, you will have to enable long paths on Windows. Run Windows Powershell as an administrator and enter the following command:
```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
You can see more information about enabling Windows long paths [here](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#enable-long-paths-in-windows-10-version-1607-and-later).

Once you confirm you're using a compatible Python version and long paths are enabled, use `pip install "tensorflow<2.11"` to install Tensorflow. Open a Python shell in the terminal by entering `python` or `python3`, whichever Python is the correct version to run Tensorflow. Enter `import tensorflow` to test if Tensorflow works. Enter `exit()` or press `Ctrl-D` to exit the Python shell. If you get an error related to the Numpy version, you may also have to downgrade your Numpy to a version less than 2.0. Enter the command `python -m pip install "numpy<2"`. Replace `python` with `python3` if you are using Python 3. Test out `import tensorflow` again and make sure it works.

More information on installing Tensorflow can be seen [here](https://www.tensorflow.org/install/pip#windows-native_1).

## 2 Support Vector Machine (SVM)

As you may recall from lecture, SVM is a suprevised learning method that can be used to separate data based on features. A support vector is a vector that essentially draws the boundaries between classes based on known, labeled data points.

### 2.1 Linearly Separable Case

First, open `svm/p1abc.py`. At the top you will notice three boolean variables, `p1a`, `p1b`, `p1c`. For now, please set `p1a` to `True`, and `p1b` and `p1c` to `False`. Make sure you are in the `lab9_2025/svm` directory within your terminal. While in the `lab9_2025` directory, enter `cd svm` to navigate to that directory. Run `p1abc.py`. You should see a figure pop up with a bunch of red data and a bunch of blue data in distinct groups. This is the known and classified data that we will use to train our first SVM.

Now, set both `p1a` and `p1b` to be `True`. This is where we actually train our data.

Within the `p1b` if statement starting on line `44`, you will see the following:
```
clf = svm.LinearSVC() # creates a Linear SVC class
clf.fit(data, val) # fits the data with and their labels (val)
                   # using an SVM with linear kernel
```
The first command makes `clf` an instance of the `LinearSVC` class and the second command uses the fit method to generate a support vector that separates the `(x,y)` data points based on their known value/classification. In this case, if you look in `svm/data_a.csv`, the red points in the bottom left corner are classified as a `0` and the blue points are classified as a `1`.

After the data has been fitted, the SVM is used to predict the classification of two additional test points, plotted with + signs. You should see that they both appear blue, meaning the SVM classified those data points as most likely belonging to the `1` label.

Now, set `p1a`, `p1b` and, `p1c` to `True` at the top and run `p1abc.py`. You should see mostly the same plot, but now with a black line running through the middle. This black line is the decision boundary determined by the SVM!

| :question: QUESTION 1 :question:   |
|:---------------------------------------------------|
| What do you think might happen if a data point were to fall exactly on the decision boundary? |

### 2.2 Nonlinear SVM

Now, open `svm/p1def.py`. At the top you will notice three boolean variables, `p1d`, `p1e`, `p1f`. For now, please set `p1d` to `True` and `p1e` and `p1f` to `False`, and run the code. This first section is for data visualization. You should see a figure pop up with a bunch of red data and a bunch of blue data in distinct groups. It is fairly obvious that a simple line will not separate the data. This is where using different kernels comes into play!

The full definition of the SVC method with its default values is as follows and can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
```
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', 
    coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
    class_weight=None, verbose=False, max_iter=-1, 
    decision_function_shape='ovr', break_ties=False, random_state=None)
```

We don't often need to deal with ALL of these values, which is why we start off by just using the defaults. In line `46` of the code we simply have `clf = svm.SVC()`. This means that all of the default parameters are used. Depending on the chosen kernel, only certain parameters are actually used by the method. For example `gamma` is not used if the kernel is linear. Don’t worry too much about this now. Check out the above link if you want to see the definition of the class and learn a little more.

Now, change `p1e` to `True` and run the code again. In this section, instead of using the `svm.LinearSVC` method, we are using the `svm.SVC` method to fit our data. By default, the `svm.SVC` method uses a radial basis function as its kernel, and it has two parameters: `gamma` and `C`. The `gamma` parameter defines how far the influence of a single training point reaches, with low values meaning far and high values meaning close. It can be seen as the inverse of the radius of influence of samples selected by the model as support vectors. 

The `C` parameter trades off correct classification of training examples against maximization of the decision function’s margin. For larger values of `C`, a smaller margin will be accepted if the decision function is better at classifying all training points correctly. A lower `C` will encourage a larger margin, and therefore a simpler decision function, at the cost of training accuracy. In other words `C` behaves as a regularization parameter in the SVM.

By default `C` is set to 1 and `gamma` is set to 'scale' which means `gamma = 1 / (nfeatures * X.var())`.

You will also see an additional data point indicated by a + sign. Although the point appears to be directly between the four clusters of data, the SVM has classified it as "blue". Why is that?

The answer can be revealed by plotting the decision boundaries! Change `p1f` to `True` and run the code again. You should see the same plot but with the decision boundaries dictated by a solid black line, and the margins dictated by dashed lines. If you recall, the goal of SVM is to find a decision boundary that maximizes the margins separating the two data sets. You'll also notice that some points are circled with green lines. These points are the support vectors - basically, the points that have the most influence on determining the location of the decision boundary. In this case, the decision boundary connects the two blue sections in the middle, while cutting off the red sections from each other.

Try changing the values of C and gamma and see what happens. Start with `gamma` values ranging from `0.1` to `10` and `C` values ranging from `0.1` to `100`. Feel free to explore other values!

| :question: QUESTION 2 :question:   |
|:---------------------------------------------------|
|  Show us some screenshots of any notable changes.  |

Now, try changing the kernel and see what happens. The following kernels are available for use: `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`. 

| :question: QUESTION 3 :question:   |
|:---------------------------------------------------|
| How well does each kernel appear to classify the data? Show some screenshots of the different kernels being used. |

Next, see what happens if we play with the polynomial kernel. By default, it is set to `degree = 3`. Let's use a higher degree and see if it helps. 

| :question: QUESTION 4 :question:   |
|:---------------------------------------------------|
| Does changing the polynomial kernel degree help? Which one appears to be the best? Is there a disadvantage to using higher degree polynomial functions? |

## 3 Neural Network (NN)

We have provided you a classic *hello world* example of NNs used to classify handwritten images of numbers to the number in the image. Make sure you are in the `lab9_2025` directory. If you are in the `lab9_2025/svm` directory, enter `cd ..` to return to the `lab9_2025` directory. Run `nn/classifier.py`.

**If you installed pytorch instead (see [1.2 Tensorflow](#12-tensorflow)) run `classifier_pytorch.py`.**

A window will show up. There are two sets of accuracy over *epochs*: one for the training data and the other for the testing data. As the epoch proceeds, we can see that the accuracies of both the training set and the test set increase as expected. Note that this is an ideal case. Overfitting could happen if the epoch number is set too high and under-fitting could happen if it is too low. Try modifying the code to use an epoch number larger than 5.

Now, take a closer look at the code, especially the comments, to understand how it works. In the [previous section](#12-tensorflow) we have installed the Tensorflow library. This is an open source library developed by Google for convenient and efficient deployment of common machine learning techniques. Keras is a NN library that is built on Tensorflow. Some background information: An alternative library is Pytorch, developed by Microsoft and Facebook; feel free to implement with both libraries and make a comparison. In 2.12, we will stick with Tensorflow.

In this lab, we use the [MNIST image set](https://paperswithcode.com/dataset/mnist), which is a set of handwritten images of numbers that are correctly labelled. Each image contains 28 × 28 pixels. The script uses 60,000 images to train our network and 10,000 to test it. The 28 × 28 pixels are converted to a single 1D array, which is then fed through the NN, where the trained `y` value is the number corresponding to the image.

Here, we use two layers of neurons, with a [*sigmoid* and a *softmax* activation function](https://medium.com/data-science/sigmoid-and-softmax-functions-in-5-minutes-f516c80ea1f9). There are a lot of other activation functions, such as *relu* and *tanh*. Give them a try and see how that changes the result. We also use *stochastic gradient decent* to find our global minimum. Alternative optimizers such as *adam* and *adagrad* are also included in Keras. A cool comparison of their performances can be found [here](https://web.archive.org/web/20220813224719/https://mlfromscratch.com/optimizers-explained/#/).

Eventually, we give each input image 10 scores based on the output of the last layer of neurons. These 10 scores are probabilities, corresponding to the numbers 0-9, and are evaluated with a mathematical technique called *cross-entropy*. The index with the highest probability is the number predicted by the image.

| :question: QUESTION 5 :question:   |
|:---------------------------------------------------|
| Why do we convert the 2-dimensional 28-28 input matrix into a 784 x 1 array? How is that different from a convolutional neural network? |

| :question: QUESTION 6 :question:   |
|:---------------------------------------------------|
| Do you notice any interesting correlation between *sigmoid* and *softmax* as activation functions? |

## 4 Convolutional Neural Network (CNN)

Open this [example notebook](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb) from TensorFlow. The notebook is on Google Colab, a hosted Jupyter Notebook service that requires no setup to use and provides free access to computing resources, including GPUs and TPUs. 

The notebook features an example of a more advanced machine learning technique that would likely be necessary to detect something like a transparent water bottle. We will use the Oxford IIIT Pet dataset, where 37 categories of pets are correctly segmented from the image and classified, and a modified version of a CNN, which has several downsampling and upsampling layers. The downsampling encoder is called MobileNetV2, detailed [here](https://arxiv.org/pdf/1801.04381.pdf), while the upsampling decoder is the `pix2pix` package from the Tensorflow example.

The code is divided into several parts. The first part involves displaying the image and the mask. As you'd expect, the image is just an image of a pet. The mask is the segmented image that we want to produce with the model. The next part is the construction of the CNN and the fitting of the system. It requires a lot more training than the previous example, 20 epochs instead of 5. The training also requires a lot more compute, which is why we asked you to run it on Colab. After each epoch, there is a callback written to save the weights at the end of the training and also a callback to display the predicted image after training. The last part is the training loss and validation loss after each epoch. There models are saved in the folder.

Click the bracket `[ ]` in front of each code block sequentially to run the code. Make sure to read the descriptions carefully. **Do not worry if you get an error during the `pip install` step (it may also take a long time). If you get an error importing the pet dataset, change the code to `dataset, info = tfds.load('oxford_iiit_pet:4.*.*', with_info=True)`.**

| :question: QUESTION 7 :question:   |
|:---------------------------------------------------|
| At which stage does *convolution* come in? |

## 5 Submission to Canvas

Please compile all your screenshots and answers in a PDF and upload it to the "Lab 9: Machine Learning" assignment on Canvas.

[^1]: Version 1 - 2020: Jerry Ng, Rachel Hoffman-Bice, Steven Yeung, and Kamal Youcef-Toumi  
  Version 2 - 2021: Phillip Daniel  
  Version 3 - 2024: Jinger Chong  
  Version 4 - 2025: Roberto Bolli, Kaleb Blake
