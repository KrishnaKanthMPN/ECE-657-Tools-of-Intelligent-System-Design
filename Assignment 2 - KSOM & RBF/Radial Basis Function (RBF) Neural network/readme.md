1. Designing a **RBF network** based on
- **Gaussian kernel** functions with constant spread function
- Using all the points in the training set as centers of the RB functions
Results-
- For the network, we have varied the value of sigma [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,14,16] and have considered all the data points as centers.
- From the plot between Testing Accuracy and Sigma, we notice that the accuracy started decreasing with the increase in the value of sigma (0.3,0.4).
- From the plot between Mean Square Error and Sigma, we notice that the error started increasing with the increase in the value of sigma (0.3,0.4).
- Smaller values of width doesn't seem to provide a good interpolation of function and larger width value, though it performs well, there could be loss of information when the ranges of the radial functions are further away from the original range of the function. Hence, we have chosen the value of sigma as 2 as it gave us the accuracy of 98.87% and better mean square error.

2. Designing a **RBF network**
- using this time only **150 centers**
- choosing the centers randomly and using **K-Means** algorithm
Results-
- The second RBF network was designed by using only 150 centres instead of all the data points. These centres are chosen in two ways
a)chosen randmoly
b)using K-Means algorithm
- K-Means from Scikit-learn was used. The spread parameter is fixed for all kernel functions, value being 2 which gave an accuracy of 98.8% in the first network implementation.


_This is my personal contribution. Please **do not** copy my solutions for academic purposes!!_
