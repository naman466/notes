

## What is a model

- Models are just mathematical functions 
- When inputs are passed, it computes an output, this is known as inference
- Equation also contains parameters, different parameter vales change the outcome of the computation
- Model equation describes a family of curves/relations which might exist between inputs and outputs, the parameters specify what that relationship is exactly

In supervised learning, we make a model that takes an input x, and outputs a prediction y. Mathematically, the model can be described as f(x) and it's output is y. 

$$ 
y = f(x)
$$
The model also contains parameters which determine this relation/equation, so to be more precise, we can write it as 

$$ 
y = f(x, \phi) 
$$
where $\phi$ is the parameter vector. 

When we train or learn a mode, we attempt at finding parameters $\phi$ that make decent predictions from the input. We learn these parameters using a training dataset of N number of input and output pairs of examples $\{x_i, y_i\}$. 

We quantify this degree of correctness/fit using a loss L. It is a scalar value which tells us how poorly our model performs for any given choice of parameters on given data. Since loss depends on only the parameters (x is always constant for any given data pair), we can treat the loss as a function of parameters ($L[\phi]$). When we train the model, we seek to find the value of $\phi$ which minimizes the value of $L[\phi]$.

$$
\hat{\phi} = \arg\min_{\phi} L(\phi)
$$

If the loss is small after finding the value of $\hat{\phi}$, we have found an appropriate value. Then, we must check on held out validation data to ensure our model generalizes to data it did not see during training phase.





