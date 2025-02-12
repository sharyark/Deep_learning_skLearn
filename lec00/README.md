# Perceptron in Deep Learning  

## What is a Perceptron?  
A **perceptron** is a type of artificial neuron used in **supervised learning**. It is the simplest model of a neural network and is used for **binary classification** (deciding between two categories).  

## How Does It Work?  
A perceptron takes multiple input values, processes them using **weights**, sums them up, and passes the result through an **activation function** to produce an output.  

### Components of a Perceptron  
1. **Inputs**: \( x_1, x_2, x_3, \dots, x_n \) (features of data)  
2. **Weights**: \( w_1, w_2, w_3, \dots, w_n \) (importance of each input)  
3. **Summation Function**: Computes \( \text{Sum} = x_1 w_1 + x_2 w_2 + x_3 w_3 + \dots + x_n w_n \)  
4. **Bias (\( b \))**: Helps adjust the output, similar to the y-intercept in a line equation  
5. **Activation Function**: Decides whether the perceptron should "fire" (output 1) or not (output 0)  

### Perceptron Formula  
\[
y = f \left( \sum_{i=1}^{n} x_i w_i + b \right)
\]  

Where **\( f \)** is the activation function.  

## Activation Function  
The simplest activation function used in a perceptron is the **step function**:  

\[
f(x) =
\begin{cases}
1, & \text{if } x \geq 0 \\
0, & \text{otherwise}
\end{cases}
\]

Other activation functions like **sigmoid**, **ReLU**, and **tanh** are used in more advanced models.  

## Learning Process  
1. Initialize weights randomly.  
2. Compute the output using the formula above.  
3. Compare the output with the actual label.  
4. Update the weights using the **Perceptron Learning Rule**:  

   \[
   w_i = w_i + \Delta w_i
   \]

   Where **\( \Delta w_i = \alpha (y_{\text{true}} - y_{\text{predicted}}) x_i \)**  
   (\( \alpha \) is the learning rate, controlling step size)  

5. Repeat the process until the model correctly classifies all training data.  

## Limitations of Perceptron  
- Can only solve **linearly separable** problems (e.g., AND, OR gates but not XOR).  
- Cannot handle **non-linear** problems (solved by Multi-Layer Perceptrons).  

## Summary  
- A perceptron is a **basic artificial neuron**.  
- It takes weighted inputs, sums them, and applies an activation function.  
- It is trained using weight updates to minimize error.  
- Works well for simple classification but has **limitations** for complex problems.  

