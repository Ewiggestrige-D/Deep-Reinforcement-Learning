# ML Lecture 23-1: Deep Reinforcement Learning
[ML Lecture 23-1: Deep Reinforcement Learning](https://www.youtube.com/watch?v=W8XF3ME8G2I)

## Scenario of Reinforcement Learning

### Agent
    - oberserve: state of 'enviroment'
    - action: change the 'enviroment'  -> reward or panelty
    - Target: agent learns to take 'ACTIONS' to 'Maximize' expected reward

### Learning to palt Go
- Supervised vs Reinforcement
  - supervised: rely on expert pre-knowledege --  human interface/label
  - reinforecement: learing from experience: try and reward -- non human interface/label

### example: playing vedio game
- widely studies:
  - gym: [URL](https://gym.openai.com)
  - Universe: [URL](https://openai.com/bug/universe)
Machine learns to play video games as human players
 - what machine observes is pixels
 - machine learns to take  proper action itself

Difficulties of Reinforcement Learing
- reward delay
  - In space invader, only 'fire' obtains reward. although the moving before 'fire' is important
  - In Go playing, it may be better to sacrifice **immediate reward** to gain more **long-term** reward
  - Agent's actions affect the subsequent data it receives
    - exploration
  
## Outline
- Policy Base: learning an **actor**
- Value Base : learning a **critic**
- Combined: **Actor+Critic**:Asychronous Advantage Actor-Critic(A3C)
  - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
### Polcy-Based Approch: Learning An Actor
Actor/Policy：$\text{Actor} = \pi (\text{Observation})$
其中Observation 是function input  
action是function output  
reward used to pick the **best function $\pi$**

**3 steps for deep learning**
1. define a set of funciton
2. goodness of function(loss)
3. pick the best function(minimize the loss)

#### **Neural Network** as Actor
- Input of neural network: the observation of machine represented as a vector or a matrix
- Output of neural network: each action corresponds to a neuron in output layer

#### **Goodne**ss of Actor
- Given an actor $\pi_{\theta(s)}$ with network parameter $\theta$
- Use the actor $\pi_{\theta(s)}$ to play the game
  - Total Reward:$R_{\theta} =\sum_{t=1}^{T}r_t$, 其中$r_t$是在timestep t获得的reward
  - Even with same actor,$R_{\theta}$ is different each time for 
    - randomness in the actor and the game
    - stochastic property
  - define $\overline{R_{\theta}}$ as the **expected value** of $R_{\theta}$
  - $\overline{R_{\theta}}$ evaluates the goodness of an actor $\pi_{\theta(s)}$ 
- An epsidode is considered as a trajectory/sequence $\tau$
  - $\tau={s_1,a_1,r_1,...,s_T,a_T,r_T}$, 其中$s_T$为时间T时候的observe的state，$a_T$为时间T时候采取的action，$r_T$是采取action 之后的reward
  - $R(\tau) = \sum_{t=1}^{T}r_t$
  - if you use an actor to play the game, each $\tau$ has a probability to be sampled:
    - probability depends on actor params $\theta$: $P(\tau|\theta)$
  - $\overline{R_{\theta}} = \sum_{\tau}R(\tau) P(\tau|\theta)$
    - Use $\pi_{\theta}$ to play game N times, obtain ${\tau^1,...,\tau^N}$ sampling $\tau$ from $P(\tau|\theta)$ N times
    - thus $\overline{R_{\theta}} = \sum_{\tau}R(\tau) P(\tau|\theta) \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n)$

#### Pick the **best function**
- Gradient Descent
  - problem statement: $\theta^{*} = arg \max_{\theta} \overline{R_{\theta}} $
  - Gradient Descent: $\theta_1 \leftarrow \theta^0 + \eta \nabla \overline{R_{\theta^0}}$
  - 其中
  $$\nabla \overline{R_{\theta}} = 
  \begin{bmatrix} 
  \frac{\partial \overline{R_{\theta^0}}}{\partial w_1} \\ 
  \frac{\partial \overline{R_{\theta^0}}}{\partial w_2} \\ 
  \dots \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial w_n} \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial b_1} \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial b_2} \\
  \dots \\
  \frac{\partial \overline{R_{\theta^0}}}{\partial b_n} \\
  \end{bmatrix}$$

  - 实际中
  $$\nabla \overline{R_{\theta}} = 
  \sum_{\tau}R(\tau) \nabla P(\tau|\theta)$$
$R(\tau)$ do not have to be differentiable
  - thus 
  $$\begin{aligned}
    \nabla \overline{R_{\theta}} & = 
  \sum_{\tau}R(\tau) \nabla P(\tau|\theta)\\
  & = \sum_{\tau}R(\tau)  P(\tau|\theta) \frac{\nabla P(\tau|\theta) }{P(\tau|\theta) } \\
  & = \sum_{\tau}R(\tau) P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) \\
  & \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau) \nabla_{\theta} \log P(\tau|\theta)
  \end{aligned}$$
  - 如何计算stochastic probability：
  $$P(\tau|\theta) =p(s_1) p(a_1|s_1,\theta)p(r_1,s_2|s_1,a_1)p(a_1|s_2,\theta)p(r_2,s_3|s_2,a_2)\dots \\
  =p(s_1) \prod_{t=1}^{T}p(a_t|s_t,\theta)p(r_t,s_{t+1}|s_t,a_t) $$

  $$logP(\tau|\theta) = \log p(s_1) +\sum_{t=1}^{T} \left( \log p(a_t|s_t,\theta) + \log p(r_t,s_{t+1}|s_t,a_t)\right) $$

  $$\nabla logP(\tau|\theta) = \sum_{t=1}^{T} \nabla \log p(a_t|s_t,\theta)$$
  
  - 化简之后
  $$\begin{aligned}
    \nabla \overline{R_{\theta}} 
  & \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau) \nabla_{\theta} \log P(\tau|\theta)\\
  & =  \frac{1}{N}\sum_{n=1}^{N}R(\tau)  \sum_{t=1}^{T} \nabla \log p(a_t|s_t,\theta)\\
  & = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} R(\tau)\nabla \log p(a_t|s_t,\theta)
  \end{aligned}$$
  - 意义：if in $\tau^n$ machine takes $a_{t}^{n}$ when seeing $s_{t}^{n}$ in
    - $R(\tau^n)$ is *positive* -> Tuning $\theta$ to **increase** $p(a_t|s_t,\theta)$
    - $R(\tau^n)$ is *negative* -> Tuning $\theta$ to **decrease** $p(a_t|s_t,\theta)$
- Add a baseline 
  为了避免$R(\tau)$都是正数，而sampling次数有限导致的sample并非全部都是最优解,可以手动减掉一个baseline值（hyper-parameter），来保证策略会保持在一个最低优化的策略上
  $$\nabla \overline{R_{\theta}} 
  = \frac{1}{N}\sum_{n=1}^{N}  \sum_{t=1}^{T} (R(\tau)-b)\nabla \log p(a_t|s_t,\theta)
  $$














# ML Lecture 23-2: Policy Gradient (Supplementary Explanation)
[ML Lecture 23-2: Policy Gradient (Supplementary Explanation)](https://www.youtube.com/watch?v=y8UPGr36ccI&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=35)

# ML Lecture 23-3: Reinforcement Learning (including Q-learning)
[ML Lecture 23-3: Reinforcement Learning (including Q-learning)](https://www.youtube.com/watch?v=2-JNBzCq77c&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=36)