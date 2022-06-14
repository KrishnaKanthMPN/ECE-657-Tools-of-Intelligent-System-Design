Here is our implementation of backpropagation algorithm. First, we initialize the weights for the network. Next, forward propogation is performed using the initialized weights and then the weights are updated using backpropagation. Prediction is done using the finalized weights.

**Parameters**:

Activation Function - Sigmoid
Learning Rate - 0.15
Output function for one-hot encoding - Max_function (user defined)
No of hidden layers - 1 (As suggested)
No of nodes in hidden layer - 5
No of epochs for training - 20, 4(final)

Once the training is done, the resultant weights are stored in A1_G55.pkl file.

test_mlp.py file contains the function which will load the weights from .pkl file and gives the predictions.

_This is my personal contribution. Please **do not** copy my solutions for academic purposes!!_
