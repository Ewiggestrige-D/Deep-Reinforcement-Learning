# Part 2 DRL Lecture 7 Sparse Reward
[DRL Lecture 7 Sparse Reward](https://www.youtube.com/watch?v=-5cCWhu0OaM)

# Sparse Reward
在特别sparse reward的情况下，只有exploration会采取可用的action，否则actor很难学习到有用的成果，因为中间步骤的变量太多，而且没有合适的reward。

## Reward Shaping
![reward shaping](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%207/reward%20shaping.png)
delicatedly designed reward to prevent hacking reward and get the final goals

- **Curiosity Reward**
[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
**ICM = Intrisinc Curiosity Module**
![Curiosity](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%207/curiosity%20reward.png)


Using $s_t$ and action $a_t$ with the nerwork module to predict target state $\hat{s}_{t+1}$,the compare the difference between $\hat{s}_{t+1}$ and the real state $s_{t+1}$ to get the **curiosity reward** $r_{t}^{i}$.

**curiosity reward** $r_{t}^{i}$. would get larger if $s_{t+1}$ is hard to predict, i.e. the *difference is larger*

> However, some state is hard to predict but not important!

Use another network as **feature extractor** and *learn the important features related to actions* .
![Intrinsic Curiosity Module](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%207/Intrinsic%20Curiosity%20Module.png)

Network2 is used to predict the appropriate action $\hat{a_{t}}$, thusly the extractor would extract the most related info to improve the reward 

## Curriculum Learning
> 按照一定的顺序（难度，复杂度，时间长短）给robot安排合适的expert experience便于学习
> Starting from simple training examples, and then
becoming harder and harder.

### Reverse Curriculum Generation
**Methods**：
1. Given a **goal state** $s_{g}$
2. Sample some states $s_{1}$ “close” to the *final goal* $s_{g}$
3. Start from states $s_{1}$, each trajectory has reward reward $R(s_1)$
4. Delete $s_{1}$ whose reward is too large (<u>already
learned</u>) or too small (<u>too difficult at this moment</u>)
5. Iteratively, Sample $s_{2s}$ from $s_{1}$, start from $s_{2}$


## Hierarchical Reinforcement Learning
[Hierarchical Reinforcement Learning with Hindsight](https://arxiv.org/abs/1805.08180)
- If *lower agent* cannot achieve the goal, the *upper agent* would get penalty.
- If an agent get to the *wrong goal*, assume the *original goal* is the wrong one.