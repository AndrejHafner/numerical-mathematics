# Homework 2 - SOR iteration method for sparse matrices

In this homework we implement the SOR iteration method for square sparse matrices.

SOR method is utilized for the problem of embedding a graph into a plane using force-directed graph drawing.
We generate Erdos-Renyi random graphs, select a number of random nodes which are fixed on a circle and the rest of the nodes are placed around them by solving the system.

Below is an example of a 2D graph embedding of a Erdos-Renyi random graph where `n = 20` and `p = 0.2`.

![](https://raw.githubusercontent.com/AndrejHafner/numerical-mathematics/hw2-sor-iteration/hw2-sor-iteration/images/er_embedding.png)

We also analyze the effect of omega parameter of the SOR method on the convergence speed.
In our setup, we generate Erdos-Renyi graphs with `n = 50` and `p = 0.5` for omega values from `0.05` to `1.0` with a step `0.5` and embed them into the plane using the above mentioned method.
We collect the number of iterations it took for the method to converge to the correct solution with tolerance `1e-6`.
This process is repeated 50 times in order to obtain the uncertainty of our estimations.

Below we show the mean of SOR convergence speed (number of iterations) at different omega values with 95% confidence interval in light blue.
We can see that the convergence speed increases with larger values of omega and that different A matrices have little to no effect on the convergence speed, as seen from the very small 95% confidence interval.
![](https://raw.githubusercontent.com/AndrejHafner/numerical-mathematics/hw2-sor-iteration/hw2-sor-iteration/images/sor_conv_speed.png)