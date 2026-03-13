# Part 2 DRL DRL Lecture 4 Q-learning (Advanced Tips)
[Lecture 4 Q-learning (Advanced Tips)](https://www.youtube.com/watch?v=2-zGCx4iv_k)

##  Tips of **Q-Learning**

### Double DQN
- Q value is usually over-estimated
![DQN value](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%204/DQN.jpeg)

- why Q-value is always **over-estimated**
$$Q(s_t,a_t) \leftrightarrow \text{Target:} \quad r_t+\max_{a}Q(s_{t+1},a)$$

Tend to select the action that is over-estimated

- Double DQN: two function Q and $Q'$
$$Q(s_t,a_t) \leftrightarrow \text{Target:} \quad r_t+Q'(s_{t+1}, arg\max_{a}Q(s_{t+1},a))$$

  - if $Q$ over-estimate action $a$, <u>**$Q'$ would give it proper credits.**</u>
  - if $Q'$ over-estimate the value, <u>**$Q$ would not be performed.**</u>


### Dueling DQN
[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- Change the **Network Architecture**
![Dueling DQN](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%204/Dueling%20DQN.png)

- output of Dueling DQN is a scalar value $V(s)$ that only relates **STATES**, and another Q-like matrix function $A(s,a)$
- To update the value output, we just need to update the scalar function $V(s)$ and do not need to re-sample all the actions again!
- **Constraint**:为了避免network将所有的scalar Function设置为0， 有如下限制
  1.  sum of $A(s,a)$ column = 0;
  2.  $V(s)$ = average of original column 
- **Implementation**: Normalize $A(s,a)$  before adding $V(s)$ (*Norm-layer*)


### Prioritized Reply

**Keypoints**：The data with **larger TD error** in previous training has **higher probability** to be sampled.

Change the uniformly distributional sampling from the experience buffer, so the network can pay more attention to the *action that deteriorate the value*

- Parameter update, procedure is also modified.
### Multi-step (Balance between MC and TD)
- Original:
$$Q(s_t,a_t) = r_t + \hat{Q}(s_{t+1},a_{t+1})$$
- **Modified**:
$$Q(s_t,a_t) = \sum_{t'=t}^{t+N} r_{t'} + \hat{Q}(s_{t+N+1},a_{t+N+1})$$

- Pro: single-step bias would be diminished
- Con: Variance increase

### Noisy Net
Reference:
- [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

- Noise on Action($\varepsilon$-greedy)
$$ a = \left\{ \begin{aligned} 
     & arg \max_{a} Q(s,a),\quad \text{with probability} 1-\varepsilon, \\
         & ramdom, \quad otherwise\\
     \end{aligned}
     \right.
$$

- Noise on **Parameters **
> Inject noise(**Gaussian Noise**) into the **parameters** of Q function at the beginning of each epoch

$$Q(s,a) \longrightarrow^{add noise} \tilde{Q}(s,a)$$
$$a= arg \max_{a} \tilde{Q}(s,a)$$

- Noise on Action
  - Given the same state, the agent may takes **different** actions. (随机尝试)
  - No real policy works in this way

- Noise on Parameters
  - Given the same (similar) state, the agent takes the **same action**.（有系统的尝试，对于params的robustness更高）
  - State dependent Exploration
  - Explore in a consistent way

### Distributional Q-function
- State action value function $Q^{\pi}(s,a)$
  - When using actor 𝜋, the **cumulated reward** expects to be obtained after seeing observation $s$ and taking $a$
![distributional Q-function](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%204/distributional%20Q-function.png)

Using the distributional action, we can choose the action has** maximize the expectation or minimize the variance**, the evaluation becomes *multi-dimentional*.
![Q-function Distribution](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%204/Q-function%20distribution.png)


### Rainbow
![Rainbow](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%204/rainbow.png)

![Rainbow-minus](/Hung-yi%20Lee/img/Part%202%20DRL%20Lecture%204/rainbow%20minus.png)

- why taking out Double DQN is **less affected**?
  - using distributional Q-function gives accurate action-value pair, and *not tend to over-estimated* which is exactly Double DQN tries to address. 


## Appendix DQN 和 Double DQN

### DQN（Deep Q-Network）

#### 1. **DQN 是什么？**

DQN 是一种将 **Q-learning** 与 **深度神经网络** 结合的 **off-policy 深度强化学习算法**，用于解决**高维状态空间**（如图像输入）下的**离散动作控制问题**。

核心思想：用一个深度神经网络 $ Q(s, a; \theta) $ 来近似最优动作价值函数 $ Q^*(s, a) $，并通过与环境交互收集经验进行训练。


#### 2. **数学公式推导**

##### (1) 背景：Q-learning 的 Bellman 最优方程

最优 Q 函数满足：
$$
Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ r(s,a) + \gamma \max_{a'} Q^*(s', a') \right]
$$

Q-learning 的更新规则（表格形式）：
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

##### (2) DQN 的挑战

直接用神经网络拟合 $ Q(s,a;\theta) $ 会遇到以下问题：
- **非平稳目标**：目标值 $ r + \gamma \max_{a'} Q(s',a';\theta) $ 随 $\theta$ 变化，导致**训练不稳定**。
- **强相关样本**：连续帧高度相关，违反 SGD 的独立同分布假设。
- **灾难性遗忘**：策略变化快，旧经验被迅速覆盖。

##### (3) DQN 的三大关键技术

为解决上述问题，DQN 引入：

###### ✅ a. **经验回放（Experience Replay）**
- 存储 transition $ (s_t, a_t, r_t, s_{t+1}) $ 到 replay buffer $ \mathcal{D} $
- 训练时随机采样 mini-batch，打破时间相关性

###### ✅ b. **目标网络（Target Network）**
- 使用另一组参数 $ \theta^- $ 构建目标网络 $ Q(s', a'; \theta^-) $
- 目标值固定一段时间，提高稳定性
- 定期软更新或硬更新：$ \theta^- \leftarrow \theta $

###### ✅ c. **损失函数（Bellman Error）**

定义 TD（Temporal Difference）误差：
$$
\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) - Q(s_t, a_t; \theta)
$$

损失函数（均方误差）：
$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

梯度更新：
$$
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right) \cdot \nabla_\theta Q(s, a; \theta) \right]
$$

> 📌 注意：这里使用的是 **off-policy** 数据（来自行为策略，如 $\varepsilon$-greedy），但目标是学习最优策略 $ \pi^* $。

---

#### 3. **DQN 在 DRL 中的作用 & 与其他价值函数的关系**

| 价值函数 | 是否使用 | 说明 |
|--------|--------|------|
| $ Q^\pi(s,a) $ | ❌ | DQN 学习的是 **最优 Q 函数 $ Q^* $**，不是某个策略 $ \pi $ 的 Q |
| $ V^\pi(s) $ | ❌ | 不显式估计 V |
| $ Q^*(s,a) $ | ✅ | 直接目标 |
| 优势函数 $ A(s,a) $ | ❌ | DQN 不使用优势函数（那是 Dueling DQN 或 A2C 的事） |

- **作用**：
  - 提供了一种端到端学习控制策略的方法（通过 $ \pi(s) = \arg\max_a Q(s,a) $）
  - 开创了“用深度网络逼近价值函数”的范式
  - 适用于**离散动作空间**（因需计算 $ \max_a Q(s,a) $）

- **局限**：
  - 无法处理连续动作（max over continuous space 不可行）
  - 存在 **Q 值过估计（overestimation bias）** 问题 → 引出 Double DQN

---

### Double DQN

#### 1. **Double DQN 是什么？**

Double DQN 是对 DQN 的改进，**解决 Q-learning 中的过估计问题**。

在标准 DQN 中：
$$
\text{Target} = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$
由于 **max 操作同时用于选择动作和评估动作**，当 Q 值有噪声时，会倾向于选择被高估的动作，导致系统性正偏差。

Double DQN 将 **动作选择** 和 **价值评估** 解耦。

---


#### 2. **数学公式推导**

##### (1) 经典 Double Q-learning 思想

维护两个 Q 网络：$ Q_1, Q_2 $

更新 $ Q_1 $ 时：
- 用 $ Q_1 $ 选动作：$ a^* = \arg\max_a Q_1(s', a) $
- 用 $ Q_2 $ 评估：$ Q_1(s,a) \leftarrow Q_1(s,a) + \alpha [r + \gamma Q_2(s', a^*) - Q_1(s,a)] $

反之亦然。

##### (2) Double DQN 的实现（巧妙利用已有目标网络）

不需要两个完整网络！而是：

- **用主网络 $ Q(\cdot;\theta) $ 选择动作**
- **用目标网络 $ Q(\cdot;\theta^-) $ 评估价值**

即：
$$
\text{Target}_{\text{DoubleDQN}} = r + \gamma \cdot Q\left( s',\ \arg\max_{a'} Q(s', a'; \theta);\ \theta^- \right)
$$

对比 DQN：
$$
\text{Target}_{\text{DQN}} = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-)
$$

> ✅ 关键区别：  
> - DQN：$ \max_{a'} Q(s',a';\theta^-) $ → 同一个网络既选动作又打分  
> - Double DQN：先用 $ \theta $ 选 $ a^* = \arg\max Q(s',a;\theta) $，再用 $ \theta^- $ 打分 $ Q(s',a^*;\theta^-) $

##### (3) 损失函数

$$
\mathcal{L}_{\text{DoubleDQN}}(\theta) = \mathbb{E} \left[ \left( r + \gamma Q\big(s', \arg\max_{a'} Q(s',a';\theta); \theta^- \big) - Q(s,a;\theta) \right)^2 \right]
$$

其余部分（经验回放、目标网络更新）与 DQN 相同。

---

#### 4. **Double DQN 在 DRL 中的作用 & 与价值函数的关系**

- **作用**：
  - 显著减少 Q 值的过估计，提升策略稳定性
  - 在 Atari 实验中表现优于 DQN
  - 成为后续算法（如 Dueling DQN、Rainbow）的标准组件

- **与价值函数关系**：
  - 仍学习 $ Q^*(s,a) $，但估计更无偏
  - 不引入新类型的价值函数（仍是 Q 函数）
  - 可与其他改进结合（如 Prioritized Replay, Dueling Architecture）

---

### 对比总结表

| 特性 | DQN | Double DQN |
|------|-----|------------|
| 目标函数 | $ r + \gamma \max_{a'} Q(s',a';\theta^-) $ | $ r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-) $ |
| 是否过估计 | 是（系统性正偏差） | 否（显著缓解） |
| 网络数量 | 1 个主网络 + 1 个目标网络 | 同左（巧妙复用） |
| 论文 | Mnih et al., Nature 2015 | van Hasselt et al., AAAI 2016 |
| 价值函数类型 | $ Q^*(s,a) $ | $ Q^*(s,a) $（更准确） |
| 动作空间 | 离散 | 离散 |
| 是否使用 $ V $ 或 $ A $ | 否 | 否 |

---

### 补充：在现代 DRL 中的地位

- DQN 和 Double DQN 主要用于 **离散动作控制**（如 Atari、棋类）
- 在 **连续控制** 中，通常使用 **Actor-Critic 框架**（如 DDPG、TD3、SAC），因为无法高效计算 $ \max_a Q(s,a) $
- Double DQN 的思想也被推广到其他算法，例如 **Double Q-learning in SAC**（使用两个 Q 网络防高估）

---

### 参考文献

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). Chapter 6 & 7.

