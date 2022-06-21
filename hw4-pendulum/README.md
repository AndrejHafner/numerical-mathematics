# Homework 4 - Simple pendulum

In the fourth homework we implemented a simple undamped pendulum (mathematic pendulum). 
Its angle $\theta(t)$ is defined with the differential equation
$$\frac{g}{l}sin(\theta(t)) + \theta''(t) = 0 $$
where $g$ is the acceleration due to Earth's gravity and $l$ is the length of the pendulum. 
The differential equation is solved using the method Runge-Kutta 4.

Below we show the pendulum swinging and show its angle and angular speed through time.
![](./images/pendulum.gif)

### Comparison of simple pendulum and harmonic oscillator
We compare the simple pendulum with the harmonic pendulum (harmonic oscillator), which has a simple solution under the restriction that the 
size of the oscillation's amplitude is much less than 1 radian. 

Under the small angle assumption that,
$$\theta \ll 1, $$
we can use the small-angle approximation for sine function,
$$sin(\theta) \approx \theta,$$
which yields the equation for harmonic oscillator,
$$\frac{g}{l}\theta(t) + \theta''(t) = 0. $$

The simple pendulum and harmonic oscillator are compared in the setting where the small angle assumption is satisfied.
The initial angle and angular speed is the same for both pendulums.
![](./images/harm_math_pendulum_comparison_small_angle.gif)
We can see that the pendulums act the same and that their speed and angle in equal in the defined timeframe (speed and angle curves overlap for each pendulum).

In the next example the small angle assumption is broken.
![](./images/harm_math_pendulum_comparison_large_angle.gif)
Although both pendulums have equal initial conditions, their further behavior starts to differ quickly. The most obvious difference can be 
seen at the points where the pendulum angular speed becomes 0 and it swings into the other direction. The harmonic oscillator stops quicker and swings 
in the other direction much faster, looking rather unnatural compared to the simple pendulum.

### Pendulum's period with respect to the initial conditions 
Finally, we analyze the period and how it changes with different initial conditions. The period depends on the initial speed which 
defines its rotational energy. In our experiment, we assume that the pendulum has a weight of 1kg and find the periods at different initial velocities.
![](./images/energy_period_plot.png)
We can see that the period increases "exponentially" (very spread out) with respect to the initial rotational energy. Further values were not computed since the pendulum starts spinning around the attachment point. 
Since the movement is not damped this goes on infinitely, therefore the period becomes infinite (if such a system is still a pendulum, which is probably not true).


