M3L
=====
M3L is a tool for multi-label classification. It also allows the user to provide a prior matrix that encodes the correlation between the labels. M3L solves an SVM-like formulation without a b, i.e, the learnt classifiers are of the form:
y=sign ( Z^t \phi (x) )
To simulate a b, we add an extra bias feature to \phi(x), thus:
\phi ( x ) = [\psi( x ) ; bias]. Look at [1] for details.

Usage

Training: The command
M3L -train [options] training_file model_file reads in training data from training_file and saves the learnt M3L classifier in model_file subject to the following options

    * -C misclassification penalty. Default 0.5
    * -a optimisation algorithm:
          o 0: Kernelised -- SMO (Default)
          o 1: Linear -- Dual co-ordinate ascent with shrinkage (choose this if you have a linear kernel)
    * -t tau. Stopping parameter. Algorithm stops when all projected gradients have magnitude less than tau. Default 0.001
    * -k kernel type (valid only with -a 0).
          o 0: Linear: x'*y
          o 1: RBF: exp(-gamma*||x-y||^2) (Default)
          o 2: Polynomial: (gamma*x'*y+u_0)^degree
    * -g gamma (kernel parameter for RBF and Polynomial kernels). Default 1
    * -d degree. Default 3
    * -r u_0. Default 1
    * -b bias. Default 1
    * -m cache size in MB. Default 1000 (the best results are obtained when the cache is large enough to hold the entire kernel matrix)
    * -R correlation matrix file. Look at example_R.txt for an example correlation matrix file (Defaults to using the identity matrix).

Testing: The command
M3L -test testing_file model_file output_file
applies the learnt M3L classifier in model_file to the test data in testing_file and saves the predictions in output_file. 

Source Files:
============
The include directory contains all the headers, and the src directory contains all the .cpp files. All the necessary constants and typedefs are
in definitions.h. myvectors.h and myvectors.cpp define vectors and matrices. structures.h and structures.cpp define other common structures and
contain routines for reading in problems and writing out models. M3LLinear.h and M3LLinear.cpp contain the largescale linear version while M3LKernel.h
 and M3LKernel.cpp contain the kernelized code. cache.h and cache.cpp, adapted from LibSVM, implement a kernel cache.

 To build from source in linux, simply type make in the terminal. 
While compiling in windows, change the line

#define WINDOWS 0

in include/definitions.h to

#define WINDOWS 1

Data format
==================
The training (and testing) problems are presented in libSVM format:

<positivelabel1>,<positivelabel2>, ... <positivelabelk> <index1>:<value1> <index2>:<value2> ...
.
.
.

Each line contains an instance and is ended by a '\n' character. 
<index>:<value> gives a feature (attribute) value.
<index> is an integer starting from 1 and <value> is a real
number. Indices must be in ASCENDING order. Labels in the testing
file are only used to calculate accuracy or errors. example_data.svm contains a sample problem.

The correlation matrix file has the following form:
0,0:<value1> 0,1:<value2> 0,2:<value3> ... <row,column>:<value>

Indices must start from 0 and the matrix must be an L X L matrix where L is the number of labels.
Look at example_R.txt for an example correlation matrix file.

Model file format
=================
For the Kernelized method, the classifier for the k'th label is given by (using the representer theorem)

y_k (x) = sign(\sum_i \theta_{ik} K(x_i, x))

We call each x_i a support vector. For each x_i there are L \theta's, where L is the number of labels.

The model file has the following format:
#Kernelized
#bias <bias>
#labels <number_of_labels>
#kernel <kernel_type>
.	|
.	|- Kernel parameters: gamma, u_0 and degree, whichever are necessary
.	|
#<number_of_support_vectors>
<List of support vectors>

Each support vector is written in a separate line. Each line has the values of the L \theta's followed by the feature vector in the same format as the data as described above. The i'th line thus has the format:
<\theta_{i1}> <\theta_{i2}> ... <\theta_{iL}> <index1>:<value1> <index2>:<value2> <index3>:<value3> ...
Look at example_model_kernel_withR.txt or example_model_kernel_withoutR.txt for an example model file for the kernelized method.


For the Linear method,
the classifier is of the form
y_k(x) = sign(Z_k^t \phi(x))
where in this case \phi(x) is merely [x; bias]. Thus, the classifier can be written as:
y_k(x) = Z_k^tx + b_k*bias
or, in vector notation:
y=Zx+b

The model file then has the following format:
#Linear
#bias <bias>
#labels <number_of_labels>
#Z
<value of the Z matrix >
#b
<value of b>

Prediction outputs
==================
Each line corresponds to a single example. Each line has the following format:
1:<value_of_label1> 2:<value_of_label2>...
The value of the label is either 1 or -1 depending on whether the label is positive or not. Look at example_predictions.txt for an example.

Running the example
===================
A toy example has been included with the source code and binaries. Unpack the code and try

> cd toyexample
> ../bin/OS/M3L -train -k 1 example_data.svm your_example_model_kernel_withoutR.txt

where OS can be one of linux, windows or windowsx64. This trains the M3L classifier with an RBF kernel on the toy data. No a priori information about label correlations is provided. Compare your learnt model to the one provided to make sure that there were no errors.

To incorporate the correlation matrix given in example_R.txt try

> ../bin/OS/M3L -train -k 1 -R example_R.txt example_data.svm your_example_model_kernel_withR.txt

Results for the linear kernel can be generated by

> ../bin/OS/M3L -train -a 1 example_data.svm your_example_model_linear_withoutR.txt 

Outputs on the test file example_test.svm can be generated by

> ../bin/OS/M3L -test example_test.svm your_example_model_linear_withoutR.txt your_example_output_linear_withoutR.txt

The Hamming Loss that should be pbtained is as follows:
Hamming Loss for label 0 : 14.000002%
Hamming Loss for label 1 : 1.000000%
Net Hamming Loss: 7.500000%


 




References
=========
1.Hariharan, B., Vishwanathan, S. V. N, Zelnik-Manor, L. and Varma, M. Large Scale Max-Margin Multi-Label Classification with Priors. In ICML, 2010.







