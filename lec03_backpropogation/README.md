# Backpropagation in a Simple Neural Network

In this network, we have:
- **2 Input Nodes**
- **1 Hidden Layer with 2 Neurons**
- **1 Output Neuron**

This gives a total of **9 trainable parameters**:
- **Hidden Layer:**  
  - Weights: 2 inputs × 2 neurons = 4 weights  
  - Biases: 2 biases (one per hidden neuron)
- **Output Layer:**  
  - Weights: 2 neurons × 1 output = 2 weights  
  - Bias: 1 bias

---

## Forward Pass

1. **Hidden Layer Computation**

   For each hidden neuron \( i \) (where \( i = 1, 2 \)):
   
   \[
   z_i = w_{i1} \cdot x_1 + w_{i2} \cdot x_2 + b_i
   \]
   
   Apply an activation function \( f \) (for example, sigmoid or ReLU):
   
   \[
   a_i = f(z_i)
   \]

2. **Output Layer Computation**

   The output neuron takes the activations from the hidden layer:
   
   \[
   z_o = w_{o1} \cdot a_1 + w_{o2} \cdot a_2 + b_o
   \]
   
   And then produces the final prediction:
   
   \[
   \hat{y} = f(z_o)
   \]

---

## Loss Function

We use the **Mean Squared Error (MSE)** loss:

\[
L = (y - \hat{y})^2
\]

Where:
- \( y \) is the true output.
- \( \hat{y} \) is the predicted output.

---

## Backpropagation: Step-by-Step Explanation

Backpropagation is the process of calculating the gradient (partial derivatives) of the loss \( L \) with respect to each trainable parameter (weights and biases) so that we know how to update them to reduce the loss.

### **Step 1: Compute the Gradient of the Loss w.r.t. the Output**

Differentiate the loss with respect to the predicted output \( \hat{y} \):

\[
\frac{dL}{d\hat{y}} = 2(y - \hat{y})
\]

This derivative tells us how much the loss changes as the predicted value changes.

---

### **Step 2: Gradients for the Output Layer**

The output is computed as:

\[
\hat{y} = f(z_o) \quad \text{with} \quad z_o = w_{o1} \cdot a_1 + w_{o2} \cdot a_2 + b_o
\]

Apply the chain rule to get the gradients:

1. **For the weights \( w_{o1} \) and \( w_{o2} \):**

   \[
   \frac{dL}{dw_{oi}} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dz_o} \cdot \frac{dz_o}{dw_{oi}} \quad \text{for } i=1,2
   \]
   
   Since:
   - \(\frac{dz_o}{dw_{oi}} = a_i\)  
   - \(\frac{d\hat{y}}{dz_o} = f'(z_o)\) (the derivative of the activation function)
   
   Thus:
   
   \[
   \frac{dL}{dw_{oi}} = 2(y - \hat{y}) \cdot f'(z_o) \cdot a_i
   \]

2. **For the output bias \( b_o \):**

   \[
   \frac{dL}{db_o} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dz_o} \cdot \frac{dz_o}{db_o}
   \]
   
   Since:
   - \(\frac{dz_o}{db_o} = 1\)
   
   We have:
   
   \[
   \frac{dL}{db_o} = 2(y - \hat{y}) \cdot f'(z_o)
   \]

---

### **Step 3: Gradients for the Hidden Layer**

For each hidden neuron \( i \) (where \( i = 1, 2 \)), the output is:

\[
a_i = f(z_i) \quad \text{with} \quad z_i = w_{i1} \cdot x_1 + w_{i2} \cdot x_2 + b_i
\]

We need to see how the loss changes with respect to \( z_i \) through the output layer.

1. **Chain Rule for Hidden Layer Weights**

   Consider weight \( w_{ij} \) for hidden neuron \( i \) coming from input \( x_j \) (where \( j = 1, 2 \)):
   
   \[
   \frac{dL}{dw_{ij}} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dz_o} \cdot \frac{dz_o}{da_i} \cdot \frac{da_i}{dz_i} \cdot \frac{dz_i}{dw_{ij}}
   \]
   
   Here:
   - \( \frac{dz_o}{da_i} = w_{oi} \) (the weight connecting hidden neuron \( i \) to the output)
   - \( \frac{da_i}{dz_i} = f'(z_i) \)
   - \( \frac{dz_i}{dw_{ij}} = x_j \)
   
   So:
   
   \[
   \frac{dL}{dw_{ij}} = 2(y - \hat{y}) \cdot f'(z_o) \cdot w_{oi} \cdot f'(z_i) \cdot x_j
   \]

2. **For the Hidden Bias \( b_i \):**

   \[
   \frac{dL}{db_i} = 2(y - \hat{y}) \cdot f'(z_o) \cdot w_{oi} \cdot f'(z_i)
   \]

---

## Parameter Update Equations

After computing these gradients, we update each parameter using **gradient descent**. For a parameter \( \theta \) (which could be any weight or bias):

\[
\theta = \theta - \eta \cdot \frac{\partial L}{\partial \theta}
\]

Where:
- \( \eta \) is the learning rate (a small constant that controls the step size).

### Updates:

- **Output Layer:**
  - \( w_{oi} = w_{oi} - \eta \cdot \left[2(y - \hat{y}) \cdot f'(z_o) \cdot a_i\right] \) for \( i = 1,2 \)
  - \( b_o = b_o - \eta \cdot \left[2(y - \hat{y}) \cdot f'(z_o)\right] \)

- **Hidden Layer:**
  - \( w_{ij} = w_{ij} - \eta \cdot \left[2(y - \hat{y}) \cdot f'(z_o) \cdot w_{oi} \cdot f'(z_i) \cdot x_j\right] \) for each \( i = 1,2 \) and \( j = 1,2 \)
  - \( b_i = b_i - \eta \cdot \left[2(y - \hat{y}) \cdot f'(z_o) \cdot w_{oi} \cdot f'(z_i)\right] \) for \( i = 1,2 \)

---

## Detailed Explanation of Each Step

1. **Forward Pass:**  
   - **Hidden Layer:** Compute each neuron's weighted sum \( z_i \) from the inputs and add the bias. Then apply the activation function \( f \) to get \( a_i \).  
   - **Output Layer:** Use the activations \( a_1 \) and \( a_2 \) to compute the output neuron's weighted sum \( z_o \), add its bias, and apply the activation function to get the final prediction \( \hat{y} \).

2. **Loss Calculation:**  
   - The loss \( L \) is measured by how far the predicted value \( \hat{y} \) is from the true value \( y \) using the formula \( (y - \hat{y})^2 \).

3. **Computing Gradients (Backpropagation):**  
   - **Output Layer Gradients:**  
     Calculate how a small change in the output (or its weights and bias) would change the loss. This is done by first differentiating the loss with respect to \( \hat{y} \), then through the activation function (via \( f'(z_o) \)), and finally how each weight affects \( z_o \) (using the hidden activations).  
   - **Hidden Layer Gradients:**  
     The gradients from the output layer are “propagated back” to the hidden layer. For each hidden neuron, we calculate how the loss changes with respect to its output \( a_i \) and then how its weighted input \( z_i \) is affected by its incoming weights and bias. This involves using the chain rule to combine the effects from the output layer and the hidden layer’s activation function derivative \( f'(z_i) \).

4. **Parameter Updates:**  
   - Once you have the gradients (the partial derivatives) for each parameter, you adjust the weights and biases by subtracting a fraction of these gradients. The fraction is determined by the learning rate \( \eta \). This step moves the parameters in the direction that decreases the loss.

5. **Iteration:**  
   - These steps (forward pass, loss calculation, backpropagation, and parameter updates) are repeated for each training example (or batch of examples) until the network learns to predict accurately.

---

This complete Markdown document summarizes both the mathematical equations and the intuitive explanation of each step in the backpropagation process for your network.
