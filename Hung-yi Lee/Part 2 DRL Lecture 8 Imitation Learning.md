# Part 2 DRL Lecture 8 Imitation Learning
[DRL Lecture 8 Imitation Learning](https://www.youtube.com/watch?v=rl_ozvqQUU8)

## Imitation Learning
**Introduction**
- imitation learning
  - also known as learning by demonstration, apprenticeship learning
- An expert demonstrates how to solve the task
  - machine can also interact with the enviroment, but cannot **explicitly obtain reward**
  - it is hard to define reward in some tasks
  - **hand-crafted rewards** can lead to **uncontrolled** behavior
- Two appoached:
  - behavior Cloning
  - inverse reinforecement learning (inverse optimal control)

### **Behavior Cloning**
- self-driving as example (similar to **Supervised Driving**)
- using expert training data $(s_i,\hat{a}_{i})$ to train a network
1.  **Problem**：expert samples only leads to very limited observation(states)
- **Dataset Aggregation**:
  - Get actor $\pi_1$ by behavior cloning
  - Using $\pi_1$ to interact with the environment
  - Ask the expert to label the observation of $\pi_1$
  - Using new data to train $\pi_{2}$
2. **Major problem**: if machine has limited capacity, it may choose the **wrong behavior** to copy.
- Some behavior must copy, but some can be ignored.
  - Supervised learning takes **all errors** equally
3. **Mismatch**: trainign data and testing data are different so that network would be **less effective**
- In supervised learning, we expect training and testing data have the same distribution.
- In behavior cloning:
  - Training: $(s,a) \sim \hat{\pi}(\text{expert})$
    - <u> **Action** $a$ taken by actor influences the distribution of $s$</u>
  - Testing: $(s',a') \sim \pi^{*}(\text{expert})$ (actor cloning expert)
    - if $\hat{\pi} = \pi^{*}$, $(s,a)$ and $(s',a')$ from the same distribution
    - if $\hat{\pi} \neq \pi^{*}$, the distribution $s$ and $s'$ can be very different

### **Inverse Reinforecement Learning** (IRL)
![Inverse Reinforcement Learning](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%208/IRL.png)
- Using the expert knowledge to manully design a **appropriate reward function**
- Using the reward function to find the **optimal actor**

**Frame of IRL**
![frame of IRL](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%208/frame%20of%20IRL.png)
- Using expert $\hat{\pi}$ to aggregate a set of experiences ${\hat{\tau}_{1},\hat{\tau}_{2},\dots,\hat{\tau}_{N}}$
- Using *actor* $\pi$ to aggregate a set of experiences ${\tau_{1},\tau_{2},\dots,\tau_{N}}$
- Obtained a reward function that complies $\sum_{n=1}^{N}R(\hat{\tau}_{N}) > \sum_{n=1}^{N}R(\tau_{N})$
- Using the reward function to evaluate and update the actor by **reinforcement learning**

**exmaples**: Parking Lot Navigation
- Reward function:
  - Forward vs. reverse driving
  - Amount of switching between forward and reverse
  - Lane keeping
  - On road vs. off road
  - Curvature of paths


### **Third Person Imitation Learning**
Ref:[Third-Person Imitation Learning](https://arxiv.org/abs/1703.01703)