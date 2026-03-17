# Part 2 DRL Lecture 6 Actor-Critic
[DRL Lecture 6 Actor-Critic](https://www.youtube.com/watch?v=j82QLgfhFiY)

## Asynchronous Advantage Actor Critic (A3C)
> Reference: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
>

### Review Policy Gradient

$$\begin{aligned}
    \nabla \overline{R_{\theta}} & = 
  \sum_{\tau}R(\tau) \nabla P(\tau|\theta)\\
  & = \sum_{\tau}R(\tau)  P(\tau|\theta) \frac{\nabla P(\tau|\theta) }{P(\tau|\theta) } \\
  & = \sum_{\tau}R(\tau) P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) \\
  & \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau) \nabla_{\theta} \log P(\tau|\theta) \\
  & =  \frac{1}{N}\sum_{n=1}^{N}R(\tau)  \sum_{t=1}^{T} \nabla \log p(a_t|s_t,\theta)\\
  & = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} R(\tau)\nabla \log p(a_t|s_t,\theta)
\end{aligned}
$$

1. Add a baseline
2. **Asign suitable credit**

$$ \nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (\sum_{t'=t}^{T}\gamma^{t'-t}\cdot r_{t'}^{n}-b)\nabla \log p(a_t|s_t,\theta)
$$

Define **Advantage Function**
$$A^{\theta}(s_t,a_t) = \sum_{t'=t}^{T}\gamma^{t'-t}\cdot r_{t'}^{n}-b$$
where 
- $\theta$ labels the ***chosen model/strategy that interacts with the environment***
- $G_{t}^{n}$ labels the *cumulated reward* obtained via interaction  (**Unstable/Random Variable**)
- $b$ labels the baseline

> Can we estimate the expected value of $G_{t}^{n}$ *cumulated reward* ?

### Review  $Q$-learning
- State value function $V^{\pi}(s)$
  - When using actor $\pi$, the *cumulated* reward expects to be obtained after visiting state $s$
- State action value function $Q^{\pi}(s,a)$
  - When using actor 𝜋, the *cumulated* reward expects to be obtained after taking $a$ at state $s$
  - 需要注意的是actor 𝜋在看见state $s$的时候不一定要采取action $a$，而$Q^{\pi}(s,a)$ 则评估的是，看见state $s$的时候强制采取action $a$所得到的accumulated reward。 接下来则让actor 𝜋按照policy自由采取动作。

### Actor-Critic
Based on definition, the expectation value of $G_{t}^{n}$ is exactly $Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n})$, which defined as 'the *cumulated* reward expects to be obtained after taking $a$ at state $s$'

$$ \mathbb{E}[G_{t}^{n}] = Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n}) 
$$

Similarly, the expectation value of $Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n})$ is $V^{\pi_{\theta}}(s_{t}^{n})$, where expectation forbids the involve of actions $a_{t}^{n}$ thus provide a proper choice of baseline 

$$
\nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} \left(Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n}) -V^{\pi_{\theta}}(s_{t}^{n}) \right)\nabla \log p(a_t|s_t,\theta)
$$

### Advantage Actor-Critic
**Advantage Actor-Critic Formula**
$$
\nabla \overline{R_{\theta,a_t}} =  \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} \left(Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n}) -V^{\pi_{\theta}}(s_{t}^{n}) \right)\nabla \log p(a_t|s_t,\theta)
$$

the corresponding  **Advantage Function**
$$ \nabla \overline{R_{\theta,a_t}} =  Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n}) -V^{\pi_{\theta}}(s_{t}^{n})
$$

Try to simplified the precedure and estimate only one network
$$\begin{aligned}
     & Q^{\pi}(s_{t}^{n},a_{t}^{n}) =\mathbb{E}[r_{t}^{n}+V^{\pi}(s_{t+1}^{n})] \\
     \rightarrow^{\text{estimate}} & Q^{\pi}(s_{t}^{n},a_{t}^{n}) = r_{t}^{n}+V^{\pi}(s_{t+1}^{n}) \\
\end{aligned}
$$
so the the *estimated avantage funtion* is that 
$$
 Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n}) -V^{\pi_{\theta}}(s_{t}^{n}) \\
 \rightarrow  r_{t}^{n}+V^{\pi}(s_{t+1}^{n})  -V^{\pi}(s_{t}^{n}) 
$$
in such case, we only need to estimate **state value function** $V^{\pi}(s_{t}^{n})$ and the variance of 
$r_{t}^{n}$ is much smaller than variance of $ Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n})$ per sa, as the $ Q^{\pi_{\theta}}(s_{t}^{n},a_{t}^{n})$  accumiulates many step reward.

![Adcantange Actor-Critc](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%206/Advantage%20Actor-Critic%20update.png)

### Tips for updating Advantage Actor-Critic
- The parameters of actor $\pi(s)$ and critic $V^{\pi}(s)$ can be shared
- Use outpit entropy as regulazition for $\pi(s)$
  - larger entropy is perferred : exploration

![shared params](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%206/shared%20params.png)

- **Asynchronous**
  1. Copy global parameters 
  2. sampling some data
  3. compute gradients **respectively**
  4. update global models 
![Asychronous update model](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%206/Asynchronous%20update.png)


## Pathwise Derivative Policy Gradient
> similar to GAN

- **Theory**

![Pathwise Derivative Policy Gradient](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%206/Pathwise%20Derivative%20policy%20gradient.png)


- **Pathwise Derivative Policy Network Gradient update models**
Action $a$ is a *continous* vector
$$a = arg \max_{a} Q(s,a)$$

Actor as the solver of this optimizartion  problem
$$
\pi'(s) = arg \max_{a}Q^{\pi}(s,a)
$$
in which action $a$ is the **output** of an actor
meanwhile, this new actor will be joined in the new large network to update the params

pipelins as follow
![Pathwise Derivative Policy Network](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%206/Pathwise%20Derivative%20policy%20Network%20update.png)

And the other tips can also be applied in the procedure,e.g. exploration, replay buffer, 

- **Difference with original Q-learning**

**Typical Q-Learning Algorithm**
1. Initialize Q-function $Q$, target Q-function $\hat{Q} =Q$ at initial
2. In each epoch, for each time step $t$\
   1. Given state $s_t$, take action $a_t$ based on $Q$ (epsilon greedy/Boltzmann Exploration)
   2. Obtain reward $r_t$, and reach new state $s_{t+1}$
   3. Store $(s_t,a_t,r_t,s_{t+1})$ into buffer
   4. Sample $(s_i,a_i,r_i,s_{i+1})$ from buffer (usually a batch)
   5. Target $y = r_i+\max_{a}\hat{Q}(s_{i+1},a)$
   6. Update the parameters of $Q$ to make $Q(s_i,a_i)$,close to Target $y$ (regression)
   7. Every $N$ steps reset $\hat{Q} \leftarrow Q$ ($N$ is a **hyperparameter**)

**Pathwise Derivative Policy Network Gradient update models**
1. Initialize Q-function $Q$, target Q-function $\hat{Q} =Q$, **actor $\pi$, target actor $\hat{\pi} = \pi$**  (直接用actor $\pi$决定而不是使用价值评价函数)
2. In each epoch, for each time step $t$
   1. Given state $s_t$, take action $a_t$ based on  **actor** $\pi$ (epsilon greedy/Boltzmann Exploration)
   2. Obtain reward $r_t$, and reach new state $s_{t+1}$
   3. Store $(s_t,a_t,r_t,s_{t+1})$ into buffer
   4. Sample $(s_i,a_i,r_i,s_{i+1})$ from buffer (usually a batch)
   5. Target $y = r_i+\hat{Q}(s_{i+1},\hat{\pi}(s_{i+1}))$ (同样直接由actor $\pi$决定)
   6. Update the parameters of $Q$ to make $Q(s_i,a_i)$,close to Target $y$ (regression)
   7. Update the parameters of $\pi$ to make $\hat{Q}(s_i,a_i)$,close to Target $y$
   8. Every $N$ steps reset $\hat{Q} \leftarrow Q$ ($N$ is a **hyperparameter**)
   9. Every $N$ steps reset $\hat{\pi} \leftarrow \pi$


### Connect With GAN
![Connect with GAN](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%206/connect%20with%20GAN.png)
