# Gradient Descent

Gradient descent is a way to find the lowest point of a function. You start at a random place on the function and take steps to go down. At each step, you look at the slope of the function to decide which way to go and how big a step to take.

Here are the key parts in simpler terms:

1. **Objective Function**: You have a function \( f(x) \) that tells you how "high" or "low" you are. You want to find the \( x \) that makes \( f(x) \) as low as possible.

2. **Gradient**: This is a fancy term for the slope or steepness of the function at a particular point \( x \).

3. **Learning Rate**: This is a number that controls how big your steps are. A small number means tiny, careful steps. A big number means large, quick steps.

4. **Algorithm Steps**: 
   - Start at a random point \( x \).
   - Find the gradient (slope) of the function at \( x \).
   - Take a step in the opposite direction of the gradient.
   - Keep doing this until you find a point that is low enough.

In mathematical terms, you update \( x \) using the formula:
\[
x_{\text{new}} = x_{\text{old}} - \text{Learning Rate} \times \text{Gradient at } x_{\text{old}}
\]
You repeat this process until the function value \( f(x) \) stops changing significantly or after a set number of steps.

That's gradient descent! It's a key tool in machine learning and other areas where you need to optimize functions.

# Momemtum

In the context of optimization algorithms, momentum is a technique used to accelerate the convergence towards the minimum of a loss function. It's particularly useful in navigating shallow, flat areas and overcoming local minima in the optimization landscape.

Here's the basic idea: In gradient descent, you update your parameters \( \theta \) by moving in the direction of the negative gradient of the loss function \( L \), scaled by a learning rate \( \alpha \). Mathematically, this is:

\[
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
\]

However, gradient descent can be slow or get stuck in local minima. Momentum aims to fix this by incorporating a fraction \( \beta \) of the previous update vector into the current update. The update rule changes to:

\[
v_{t+1} = \beta v_t + (1 - \beta) \nabla L(\theta_t)
\]
\[
\theta_{t+1} = \theta_t - \alpha v_{t+1}
\]

Here, \( v_{t+1} \) is the velocity (momentum term) at time \( t+1 \), and \( \beta \) is a hyperparameter between 0 and 1 (often set to values like 0.9). This has the effect of smoothing out the updates. If the gradient keeps pointing in the same direction, the momentum term \( v \) will accumulate and result in faster convergence. If the gradient changes direction, the momentum term helps dampen the oscillations.

The inclusion of momentum effectively gives the optimization "memory" of past gradients, allowing it to avoid oscillations and navigate more smoothly towards the global (or a good local) minimum. This often results in faster and more stable convergence in training algorithms like neural networks.