
**Dataset**:
The IMDB dataset contained 25k train and 25k test sets. Postive and negative reviews for both train and test datasets were present in separate folders.

**Architecture**:
![image](https://user-images.githubusercontent.com/14235791/172992001-20e3a367-3848-4736-93ae-2382b5e8ff0a.png)
- The embedding layer encodes the input sequence into a sequence of dense vectors of dimension mentioned.
- Dropout - This is a regularization method where input and recurrent connections to LSTM units are probabilistically excluded from activation and weight updates while training a network. This has the effect of reducing overfitting and improving model performance. Dropout rate of 0.2 has been used.
- We considered 100 LSTM units for the model
- Relu - Rectified linear unit function will output the input if it is positive, otherwise it will output zero. This overcomes the vanishing gradient problem, allowing models to learn faster and perform better.
- Sigmoid - This function limits the output to a range between 0 and 1.
- We are using "Adam" optimizer as it handles sparse gradients and trains.
- Loss as 'binary_crossentropy' is used as we have only two label classes.
