

## Models and Learning Them

- Models are mathematical functions 
- When inputs are passed, it computes an output, this is known as inference
- Equation also contains parameters, different parameter values change the outcome of the computation
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

When we train or learn a model, we attempt at finding parameters $\phi$ that make decent predictions from the input. We learn these parameters using a training dataset of N number of input and output pairs of examples $\{x_i, y_i\}$. 

We quantify this degree of correctness/fit using a loss L. It is a scalar value which tells us how poorly our model performs for any given choice of parameters on given data. Since loss depends on only the parameters (x is always constant for any given data pair), we can treat the loss as a function of parameters - $L(\phi)$. When we train the model, we seek to find the value of $\phi$ which minimizes the value of $L(\phi)$.

$$
\hat{\phi} = \arg\min_{\phi} L(\phi)
$$

If the loss is small after finding the value of $\hat{\phi}$, we have found an appropriate value. Then, we must check on held out validation data to ensure our model generalizes to data it did not see during training phase.

## Example - Linear Regression


A 1D linear regression model uses the straight line to describe the relation between x and y

$$
\begin{align*}
\hat{y} &= f(x, \phi) \\
&= \phi_1 + \phi_2 x
\end{align*}
$$


This equation can describe infinitely many lines, to find our best fitting model, we need to minimize the loss i.e. the distance between the predictions $\hat{y}$ as described by our model and the actual y. To do this, we define our loss function as:

$$
\begin{align*}
\ell(\phi)_i &= (y_i - \hat{y_i})^2
\end{align*}
$$

Averaging this over the entire dataset, we get 
$$
\begin{align*}
L(\phi) &= \frac{1}{n} \sum_{i=1}^n \ell (\phi)_i\\
L(\phi) &= \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2 \\

&= \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i, \phi))^2
\end{align*}
$$

Where n is the number of samples. This is commonly known as the mean square error loss. The square makes it so that the deviation of the line (above or below the points) is irrelevant. 

Now, we need to find $\hat{\phi}$ such that:

$$
\begin{align*}
\hat{\phi} &= \arg\min_\phi L(\phi) \\
&= \arg\min_\phi \left[
\frac{1}{n} \sum_{i=1}^n (y_i - f(x_i, \phi))^2 
\right] \\
&= \arg\min_\phi \left[
\frac{1}{n} \sum_{i=1}^n (y_i - (\phi_1 + \phi_2 x))^2 
\right] \\
\end{align*}
$$The process of finding $\hat{\phi}$ is called fitting, training or learning. The most basic method is to initialize the parameters randomly, and walk downhill on the loss curvature (towards the minimum) until no further convergence is possible.

Now that we have the trained model, we can test it on the held out validation data. 

A model that captures training data very well, but does not perform well on testing data (it memorizes the training data due to it's expressive capability) is called "overfit" and a model which is too simple to the patterns of the data is called "underfit"