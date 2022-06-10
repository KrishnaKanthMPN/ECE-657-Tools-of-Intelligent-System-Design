
**Dataset**: Dataset consists of 60,000 32x32 colour images belonging to 10 classes, with 6,000 images per class. CIFAR10 Dataset (Canadian Institute For Advanced Research and 10 denotes the number of classes present in the dataset)

**Architecture**: 
We have built a Convolutional neural network with the following architecture:
2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function
2x2 Max pooling layer
2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function
2x2 Max pooling layer
Fully connected layer with 512 units and a sigmoid activation function
Dropout layer with 0.2 dropout rate
Fully connected layer with 512 units and a sigmoid activation function
Dropout layer with 0.2 dropout rate
Output layer with **Softmax** activation function and **10 neurons** (since we have 10 different classes in our dataset)

**Summary**: The above CNN network with droputs and maxpool layers added in its architecture performed well with no overfit. In the dropout layer, some number of neurons are randomly ignored or “dropped out". As there are lesser weights associated to update during backpropagation, time taken for training reduced. The time taken by the network for training is 49 sec which is better than other network designs performed.

**Future Scope/Our recommendations to improve the network**:
Considering the above architecture as it has provided better performance.
1. Number of epochs can be increased.
2. Only 20% of actual training data is considered. Considering the whole 50,000 images for training will definitely help in getting better accuracies.
3. For images with subject not in center, padding has to be introduced so as to preserve the border features.
4. Data Augmentation can be introduced. This is a process of generating more images(shifted image, tilted image, rotated image etc) from a single
image. This helps in reducing the overfit problem and can help increase the accuracy.
5. When data augmentation is considered, the model can be deeper to extract the features, so additional layers can be introduced as well.
