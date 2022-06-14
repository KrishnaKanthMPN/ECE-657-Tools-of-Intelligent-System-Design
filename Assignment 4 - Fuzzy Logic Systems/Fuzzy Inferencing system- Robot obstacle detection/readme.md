Created a Fuzzy inferencing system to decide the Speed and steering in order to avoid obstacles.

We have created the **rule base** for fuzzy system. We made sure that the guide given below was included.
- When an obstacle is detected, avoid it: does not matter which direction; left or right
- When there is no obstacle move forward
- Speed is reduced when turning away from an obstacle
- Speed is increased when cruising (when there is no obstacle
![RuleBase](https://user-images.githubusercontent.com/14235791/173474934-07942939-51ef-4087-af84-28a29befc348.JPG)

Here,
-> The **inferencing system** that will be used is **Mamdani**.
-> The **defuzzification** method used is **Mean of Maxiums(MOM)**.

- Mamdani fuzzy inference systems are used for MISO (Multiple Input Single Output) and MIMO (Multiple Input Multiple Output) systems. Sugeno fuzzy inference systems are used for MISO (Multiple Input Single Output) systems.

- We have to design an inference system with two antecedents (distance, angle) and two consequents (speed, steering turn) - a MIMO system and hence we are considering Mamdani system for implementation.

- The scikit-fuzzy controller is by default a mamdani system.

- The Centre of Area(CoA) defuzzification method would be used ideally in most of the scenarios. However, this is not an ideal method for a robot as it can lead to producing bad output values. For example, If the robot wants to move straight ahead and there is an obstacle in front, the robot can either move right or left. The output fuzzy set in this case would have two peaks – one on the right and the other on left. If COA method was used for defuzzification, an output which would take the robot even closer towards the target would be produced. To overcome this problem, another technique mean of maximums(mom) is considered here.
