We have explored two Recurrent neural networks - Long Short-Term Memory (LSTM) and Gated recurrent units (GRU). We havedone performance evaluation on both these RNN's and have finalized with LSTM due to better training and less loss incurredduring our training.

- Following is the **architecture** of our **LSTM network** -
1. LSTM Layer 1: Units/Neurons = 200, Activation fucntion- "tanh", Recurrent Activation function- "Sigmoid"
2. Dropout Layer 1: 20% dropout
3. LSTM Layer 2: Units/Neurons = 200, Activation fucntion- "tanh", Recurrent Activation function- "Sigmoid"
4. Dropout Layer 2: 20% dropout
5. Output Layer: Outputs the predicted stock price (o/p dimension = 1)

- **Dataset**: The dataset given (q2_dataset.csv) has 4 features for each date. We are required to convert this dataset by using the latest 3 days as thefeatures and the next day’s opening price as the target.
- We have created a dataframe with Open, High, Low and Volume columns for all the 3 days. Assigned the feature values of timestampst-1, t-2, and t-3 to the 1st sample data point in the new dataframe.
- Final Dataset dimension = 1256 rows × 13 columns

- **Optimizer** used is "Adam" so as to handle sparse gradients and to train the nework efficiently.
- **Loss function** choosen is "Mean Squared Error".
- Number of epochs = 1000, Batch size = 64
(We have finalized these values after evaluating the loss and the performance of other combinations).

- **Output of the Training loop**: The loss encountered during the training phase of the model is 8.6

- **Output from Testing:** The loss encountered during the testing phase of the model is 8.7. The output plot of True vs Predicted values clearly depicts that majority of the predictions made by the finalized LSTM model areaccurate. The predicted values lags behind the actual/true value when there are spikes in the prices. Apart from that, the model couldreplicate the upward and downward trends in the Opening price prediction.
- ![image](https://user-images.githubusercontent.com/14235791/172987865-0a04e78d-1aae-4d98-aa9d-8236ccbf3e9b.png)
