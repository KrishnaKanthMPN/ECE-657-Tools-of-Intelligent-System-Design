Designing a **Kohnen Self Organizing Map**
- The output of KSOM has to be some shades of color mapped over 100 by 100 grid of neurons.
- The training input of the KSOM are 24 colors (shades of Red, Blue, Green, Yellow, Tale, Pink).
- The shades were picked from https://www.rapidtables.com/web/color/RGB_Color.html

**Algorithm steps**

First, the weights of the nodes are initialized to a random value.
Euclidean distance between the input data and the associated weight vector are found.
The node that produces the smallest distance is considered as winning node.
The weights of winning neuron and its neighbours are updated.
This is done until the last epoch.
Function implementing this algorithm has been designed. Output of this is shades of colors mapped over a 100 by 100 grid. Figures of SOM after 20, 40, 100, 1000 epochs changing the value of 𝜎0 = 1, 10, 30, 50, 70 has been generated. The learning rate and neighbourhood radius are decreased over time as mentioned in the question.

**Results**-
From the figures, it is observed that for the 20 epochs with a radius 1, there were small clusters formed. The clusters are far apart even after 1000 epochs for radius 1.
Slowly, the clusters size started increasing with an increase in the radius which means the algorithm started understanding the patterns and have started mapping the input colors on to the grid. At the end of 1000 epochs for radius 70 almost all the training data points are clustered into different colors.
Increase in the radius helped in proper cluster formation and increase in the number of epochs helped in clustering all the pixels. This shows the algorithm performed well in clustering the colors chosen.
