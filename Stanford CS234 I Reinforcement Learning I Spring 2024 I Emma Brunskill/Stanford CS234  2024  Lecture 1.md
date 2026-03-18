# Stanford CS234 Reinforcement Learning | Introduction to Reinforcement Learning | 2024 | Lecture 1
[Stanford CS234 Reinforcement Learning | 2024 | Lecture 1](https://www.youtube.com/watch?v=WsvFL-LjA6U)

## Overview of Reinforcement Learning 

> Idea of automatic agent learning:
> Learning through experience/data to make good decisions under uncertainty

**Example**: ChatGPT
![ChatGPT](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%201/ChatGPT.jpeg)
1. step 1: behvior cloning/imitation learning 
2. step 2: Model based reinforcement learning: use a preferenced date to learn a preferenced model
3. step 3: Reinforcement Learning from Human Feedback(RLHF)

### **Reinforcement Learning (Generally) Involves**
- Optimization
  - Goal is to find an optimal way to make decisions
    - Yielding best outcomes or at least very good outcomes
  - Explicit notion of decision utility
  - Example: finding minimum distance route between two cities given network of roads
  - “hypothesis: that the generic objective of maximising reward is enough to drive behaviour that exhibits most if not all abilities that are studied in natural and artificial intelligence.” – “**Reward is enough**” Silver, Singh, Precup, Sutton
- Delayed consequences
  - Decisions now can impact things much later
    - Saving for retirement
  - Introduces two challenges
    - When **planning**: decisions involve reasoning about not just immediate benefit of a decision but also its longer term ramifications
    - When **learning**: temporal credit assignment is hard (what caused later high or low rewards?)
- Exploration
  - Learning about the world by making decisions
    - Agent as scientist
  - Decisions impact what we learn about
    - Only get a reward for decision made
    - Don’t know what would have happened for other decision
- Generalization
  - Policy is mapping from past experience to action

### Two of the Problem Categories Where RL is Particularly Powerful

1. **No examples of desired behavior**: e.g. because the goal is to go
beyond human performance or there is no existing data for a task.
2. Enormous **search** or **optimization** problem with *delayed outcomes*

### Sequential Decision Making
![Sequential Decision Making](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%201/Sequential%20Decision%20Making.png)
- Goals: Select actions to maximize total expected future reward
  - May require balancing immediate & long term rewards
- Each time step $t$:
  - Agent takes an action $a_t$
  - World updates given action $a_t$ , emits observation $o_t$ and reward $r_t$
  - Agent receives observation $o_t$ and reward $r_t$

**History: Sequence of Past Observations, Actions & Rewards**
- History $h_t = \{a_1,o_1,r_1,\dots,a_t,o_t,r_t\}$
- Agent chooses action based on history
- State is information assumed to determine what happens next
  - Function of history: $s_t = f(h_t)$

### Markov Assumption
1. Information state: sufficient statistic of history
2. State $s_t$ is ***Markov*** if and only if:
$$
p(s_{t+1}|s_t,a_t) = p(s_{t+1}|h_t,a_t)
$$
3. Future is independent of past given present state
> Is there a difference between *state* and *oberservation* in such case?
> Prof Brunskill: Yes! The **state** defined in Atari game, by Deepmind, was the last 4 frames, not one! It gives you the info of velocity and acceleration.

### Why is Markov Assumption Popular?
- Simple and often can be satisfied if include **some history** as part of the state
- In practice often assume **most recent observation** is sufficient statistic of history: $s_t = o_t$
- State representation has big implications for:
  - Computational complexity
  - Data required
  - Resulting performance (there will be trade-offs between bias and variances.)
> More specificly, there will be trade-offs between usding states that are really small and easy for us to work with, but are not realy able to capture the complexity of the world and the applications we cared about. So that it might be fast to learn with those sort of representations, but ultimately, performance is poor

### Types of Sequential Decision Processes
- Is state Markov? Is world partially observable? (POMDP)
- Are dynamics deterministic or stochastic?
- Do actions influence only immediate reward (bandits) or reward and next state ?

**Example: Mars Rover as a Markov Decision Process**

![Mars Rover](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%201/Mars%20Rover.png)

- State: Location of rover $(s_1, . . . , s_7)$
- Actions: TryLeft ot Try Right
- Rewards:
  - +1 in state $s_1$
  - +10 in state $s_7$
  - 0 in all other states

### MDP Model
- Agent’s representation of how world changes given agent’s action
- Transition / dynamics model predicts next agent state
$$
p(s_{t+1}=s' | s_t=s, a_t=a)
$$
- Reward model predicts immediate reward
$$
r(s_t =s, a_t =a) = \mathbb{E}[r_t | s_t=s, a_t=a ]
$$

### **Policy**
- Policy $\pi$ determines how the agent chooses actions
- $\pi : S \rightarrow A$, mapping from from states to actions
- Deterministic policy:
$$\pi(s) =a $$
- Stochastic policy: (**Probable Distribution**)
$$\pi(s|a) =P(a_t=a | s_t =s) $$


**Evaluation and Control**
- Evaluation
  - Estimate/predict the expected rewards from following a given policy.(define how good the policy is )
- Control
  - Optimization: find the best policy 

## Making Sequences of Good Decisions Given a Model of the World
- Assume finite set of states and actions
- Given models of the world (dynamics and reward)
- Evaluate the performance of a particular decision policy
- Compute the best policy
- This can be viewed as an AI planning problem

### Outlines
- Markov Processes
- Markov Reward Processes (MRPs)
- Markov Decision Processes (MDPs)
- Evaluation and Control in MDPs

### Markov Process or Markov Chain
- Memoryless random process
  - Sequence of random states with Markov property
- Definition of **Markov Process**
  - $S$ is a (finite) set of states ($s \in S$)
  - $P$ is dynamics/transition models that specofies $P(s_{t+1} =s'|s_t =s)$
- Note: no rewards, no actions
- If finite number (N) of states, can express P as a matrix
$$
P = \begin{pmatrix}
    P(s_1|s_1) & P(s_2|s_1) & \dots & P(s_N|s_1) \\
    P(s_1|s_2) & P(s_2|s_2) & \dots & P(s_2|s_N) \\
    \vdots & \vdots & \ddots &\vdots \\
    P(s_1|s_N) & P(s_2|s_N) & \dots & P(s_N|s_N)
\end{pmatrix}
$$

**Examples**
![Mars Rover Markov Chain ](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%201/Mars%20Rover%20Markov%20Chain%20Transition%20Matrix.png)
Transition Matrix $P$
$$
P = \begin{pmatrix}
0.6 & 0.4 & 0 & 0 & 0 & 0 & 0 \\
0.4 & 0.2 & 0.4 & 0 & 0 & 0 & 0 \\
0 & 0.4 & 0.2 & 0.4 & 0 & 0 & 0 \\
0 & 0 & 0.4 & 0.2 & 0.4 & 0 & 0 \\
0 & 0 & 0 & 0.4 & 0.2 & 0.4 & 0 \\
0 & 0 & 0 & 0 & 0.4 & 0.2 & 0.4 \\
0 & 0 & 0 & 0 & 0 & 0.4 & 0.6 \\
\end{pmatrix}
$$

- Example: Sample episodes starting from S4
  - $s_4, s_5, s_6, s_7, s_7, s_7, \dots$
  - $s_4, s_4, s_5, s_4, s_5, s_6, \dots$
  - $s_4, s_3, s_2, s_1,\dots$
  
### Markov Reward Processes (MRPs)s
- Markov Reward Process is a **Markov Chain + rewards**
- Definition of **Markov Reward Process** (MRP)
  - $S$ is a (finite) set of states ($s \in S$)
  - $P$ is dynamics/transition models that specofies $P(s_{t+1} =s'|s_t =s)$
  - $R$ is a reward function $R(s_t=s)=\mathbb{E}[r_t|s_t=s]$
  - Discount factor $\gamma \in [0,1]$
- Note: no actions 
- If finite number(N) of states, can espress $R$ as a vector


### Return & Value Function
- Definition of Horizon ($H$) 
  - Number of time steps in each episode
  - Can be infinite
  - Otherwise called **finite Markov reward process**
- Definition of Return, $G_t$ (for a Markov Reward Process)
  - Discounted sum of rewards from time step $t$ to horizon $H$, which is the end of such *eposide*
  $$
  G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dotsb + \gamma^{H-1} r_{t+H-1}
  $$
- Definition of State Value Function, $V(s)$ (for a **Finite** Markov Reward Process)
  - Expected return from starting in state $s$
  $$ V(s) =\mathbb{E}[G_t|s_t=s] =\mathbb{E}[ r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dotsb + \gamma^{H-1} r_{t+H-1}|s_t=s]
  $$
    > Generally,  $V(s)$ is not same with the $G_t$ since the action be taken is a *variable* which leads to different reward, unless the processes are ***deterministic***
### Discount Factor
- Mathematically convenient (avoid infinite returns and values), perticularly when we have **infinite steps**
- Humans often act as if there’s a discount factor $< 1$
- If episode lengths are always finite ($H < \infty$), can use $\gamma =1$
- $\gamma =0$: Only care about **immediate reward**
- $\gamma =1$: **Future reward** is as beneficial as *immediate reward*