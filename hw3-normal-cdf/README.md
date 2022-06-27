# Homework 3 - Normal CDF calculation using Adaptive Simpson method

In this homework we implement the calculation of Normal CDF for $X \sim N(0, 1)$ using the Adaptive Simpson numerical
integration method.

Function is defined as:

$$
\Phi(x) = P(X \leq x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{-\frac{t^2}{2}}dt
$$

We calculate the value of the function with a relative accuracy of $10^{-10}$.
The lower bound of the integral is chosen in such a ways to ensure the desired relative accuracy.
Bound is selected by creating an interval of values in log space and calculating the difference between consecutive function values on that interval. When the absolute difference is below the desired epsilon value, we choose the value as the lower bound. This is calculated once per epsilon value and cached in order to decrease computational complexity.

## Performance

We test the performance of our method at estimating the value of Normal CDF at different desired accuracy levels.
We test 15 epsilon (accepted error) values from $10^{-1}$ to $10^{-15}$, where we repeat the calculation 50 times at each step. 
On the log-log plot below we can see that computation time increases exponentially with lower values of epsilon. This is because the Adaptive Simpson method has to increase the depth of recursion (when calculating the split $h$ to ensure desired accuracy) for each increase of accuracy which in turn leads to exponential increase of subintervals on which Simpson method is evaluated.

![](https://raw.githubusercontent.com/AndrejHafner/numerical-mathematics/hw3-normal-cdf/hw3-normal-cdf/performance.png)


