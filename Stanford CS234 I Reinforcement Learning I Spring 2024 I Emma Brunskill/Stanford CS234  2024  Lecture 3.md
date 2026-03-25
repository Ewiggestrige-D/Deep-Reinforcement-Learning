# Stanford CS234 Reinforcement Learning | Introduction to Reinforcement Learning | 2024 | Lecture 3
[Stanford CS234 Reinforcement Learning | 2024 | Lecture 3](https://www.youtube.com/watch?v=jjq51TRNVvk)


# Lecture 3: Model-Free Policy Evaluation: Policy Evaluation Without Knowing How the World Works

> - In a tabular MDP asymptotically value iteration will always yield a policy with the same value as the policy returned by policy iteration 
> Answer: True. Both are guaranteed to converge to the optimal value function and a policy with an optimal value
>
>
> - Can value iteration require more iterations than $|A|^{|S|}$ to compute theoptimal value function? (Assume |A| and |S| are small enough that each round of value iteration can be done exactly).
> Answer: True. As an example, consider a single state, single action MDP where $r (s, a) = 1$, $\gamma = .9$ and initialize $V_0(s) = 0, V^*(s) = \frac{1}{1-\gamma} $but after the first iteration of value iteration, $V_1(s) = 1$


## Evaluation through Direct Experience
- Estimate expected return of policy $\pi$
- Only using data from environment (direct experience)
- Why is this important?
- What properties do we want from policy evaluation algorithms?


**Policy Evaluation**
- Estimating the expected return of a particular policy if don’t have access to true MDP models
- Monte Carlo policy evaluation
  - Policy evaluation when don’t have a model of how the world works
    - Given on-policy samples
- Temporal Di↵erence (TD)
- Certainty Equivalence with dynamic programming
- Batch policy evaluation

### Recall : definition
- Definition of Return, $G_t$ (for a MRP)
  - Discounted sum of rewards from time step $t$ to horizon
  $$G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^n r_{t+n} + \dots$$

- Definition of ***State Value Function***, $V^{\pi}(s)$
  - Expected return starting in state $s$ under policy $\pi$
  $$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t|s_t = s] =  \mathbb{E}_{\pi}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^n r_{t+n} + \dots|s_t = s] $$

- Definition of ***State-Action Value Function***, $Q^{\pi}(s)$
  - Expected return starting in state $s$, taking action $a$ and following policy $\pi$
  $$\begin{aligned}
    Q^{\pi}(s) & = \mathbb{E}_{\pi}[G_t|s_t = s,a_t = a]  \\
    & =  \mathbb{E}_{\pi}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^n r_{t+n} + \dots|s_t = s,a_t = a]
  \end{aligned}$$


  ### Recall : Dynamic Programming for Policy Evaluation
  - In a Markov decision process
  $$\begin{aligned}
    V^{\pi}(s) &= \mathbb{E}_{\pi}[G_t|s_t = s] \\
    & = \mathbb{E}_{\pi}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^n r_{t+n} + \dots|s_t = s] \\
    & =R(s,\pi(s)) + \gamma \sum_{s' \in S}P(s'|s,\pi(s)) V^{\pi}(s') 
  \end{aligned}$$
    - we use $\sum_{s' \in S}P(s'|s,\pi(s)) V^{\pi}(s') $ to substitude $\mathbb{E}_{\pi}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^n r_{t+n} + \dots|s_t = s]$
    - This substitution is an instance of ***bootstrapping***
  - If given dynamics and reward models, can do *policy evaluation* through dynamic programming
  $$V^{\pi}_{k}(s) = r(s,\pi(s)) + \gamma \sum_{s' \in S}P(s'|s,\pi(s)) V_{k-1}^{\pi}(s')  $$

    - Note: before convergence, $V^{\pi}_{k}(s)$ is an estimate of $V^{\pi}(s)$

---
在 **Q-learning** 中，**bootstrap（自举）** 是指：**使用当前对后续状态价值的估计来更新当前状态-动作对的价值估计**。这是一种“用估计值来更新估计值”的机制，是时序差分（Temporal Difference, TD）方法的核心特征。


#### 🔍 什么是 Bootstrap？直观理解

想象你在爬山，想知道山顶有多高。你没有 GPS，但你知道：
- 你现在的位置海拔是 100 米；
- 前方 1 公里处有个瞭望台，你**估计**那里的海拔是 200 米。

于是你**用这个估计值（200 米）来修正你对自己当前位置价值的理解**——比如：“既然去瞭望台能到 200 米，那我现在这位置应该比 100 米更有潜力”。

这就是 **bootstrap**：**用未来的（估计的）价值来更新现在的价值**，而不是等到真正到达山顶（最终结果）才学习。

---

#### 📐 Q-learning 中的 Bootstrap 数学表达

Q-learning 的更新公式为：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

其中：
- $ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) $ 是 **TD target（时序差分目标）**
- 关键点在于：**$\max_{a} Q(s_{t+1}, a)$ 是对下一状态最优价值的当前估计**，而不是真实值，也不是等 episode 结束后才知道的完整回报。

✅ 这个 **用 $Q(s_{t+1}, \cdot)$ 的估计值来构造目标** 的过程，就是 **bootstrap**。

---

#### ⚖️ 对比：有 bootstrap vs 无 bootstrap

| 方法 | 是否 bootstrap | 如何更新 |
|------|----------------|--------|
| **Q-learning / SARSA** | ✅ 是 | 用下一步的 Q 值估计来更新当前 Q 值（单步或 n 步） |
| **Monte Carlo (MC)** | ❌ 否 | 等整个 episode 结束，用真实 return $G_t = \sum \gamma^k r_{k}$ 更新，不依赖未来估计 |

- **Bootstrap 的优点**：可以在线学习（online），每一步都能更新；方差低。
- **Bootstrap 的缺点**：引入偏差（bias），因为用的是估计值而非真实值；可能传播错误（尤其在函数逼近时导致不稳定，如 DQN 需要经验回放和目标网络来缓解）。


#### 🧠 为什么叫 “bootstrap”？

这个词源于英文谚语 “pull oneself up by one’s bootstraps”（拽着自己的鞋带把自己提起来）——看似不可能，却通过自我迭代逐步提升。  
在 RL 中，agent **从零开始，仅靠自身不断改进的估计值，逐步逼近真实价值函数**，正是这种“自举”精神的体现。


### ✅ 总结

> **Q-learning 中的 bootstrap，是指在更新 $Q(s_t, a_t)$ 时，使用对下一状态最优 Q 值的当前估计 $\max_a Q(s_{t+1}, a)$ 来构建学习目标。这是 TD 方法的核心机制，使其区别于 Monte Carlo 方法，并支持单步、在线学习。**

这也是为什么 Q-learning 属于 **model-free + bootstrapping + off-policy** 的经典算法。

---



## Monte Carlo (MC) Policy Evaluation
> for most of today assume **deterministic policy**
> The **Value** of this function is the **average return**
$$G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^{T_i -t} r_{T_i}
$$ in MDP under policy $\pi$
- Expectation over trajectories $\tau$ generated by following $\pi$
$$ V^{\pi}(s) =\mathbb{E}_{\tau \sim \pi}[G_t|s_t = s] $$
- If trajectories are all finite, sample set of trajectories & average returns
  - Note: all trajectories may not be **same length** (e.g. consider MDP with terminal states)
  - Does not require MDP dynamics/rewards
  - Does not assume state is Markov
  - Can be applied to episodic MDPs
    - Averaging over returns from a complete episode
    - Requires each episode to terminate(Finite)


### First-Visit Monte Carlo (MC) On Policy Evaluation
Initialize $N(s) = 0, G(s) = 0,\quad \forall s\in S$
Note: $N$ is the counts of times that we have updated esdtimate of state $s$. It would be different for different MC sample
Then **Loop**
- Sample episode $i = \{s_{i,1},a_{i,1},r_{i,1},s_{i,2},a_{i,2},r_{i,2},\dots, s_{i,T_i},a_{i,T_i},r_{i,T_i}\}$
- Define $G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^{T_i -t} r_{T_i} $ as return form time step $t$ onwards in $i$th episode
- For each time step $t$ until $T_i$ ( the end of the episode $i$ )
  - If this is the **first** time $t$ that state $s$ is visited in episode $i$
    - Increment counter of total ***first visits***: $N(s) = N(s) + 1$
    - Increment total return $G(s) = G(s) + G_{i ,t}$
    - Update estimate $V^{\pi}(s) = G(s)/N(s)$

### Every-Visit Monte Carlo (MC) On Policy Evaluation
Initialize $N(s) = 0, G(s) = 0,\quad \forall s\in S$
Then **Loop**
- Sample episode $i = \{s_{i,1},a_{i,1},r_{i,1},s_{i,2},a_{i,2},r_{i,2},\dots, s_{i,T_i},a_{i,T_i},r_{i,T_i}\}$
- Define $G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^{T_i -t} r_{T_i} $ as return form time step $t$ onwards in $i$th episode
- For each time step $t$ until $T_i$ ( the end of the episode $i$ )
  - State $s$ is visited at time $t$ in episode $i$
  - Increment counter of ***total visits***: $N(s) = N(s) + 1$ 
  - Increment total return $G(s) = G(s) + G_{i ,t}$
  - Update estimate $V^{\pi}(s) = G(s)/N(s)$

### Incremental Monte Carlo (MC) On Policy Evaluation
> MAitain a running estimate for what is the value under a policy for a particular state
>  And you smoothly update that as you get more data



After each episode $i = \{s_{i,1},a_{i,1},r_{i,1},s_{i,2},a_{i,2},r_{i,2},\dots, \}$
- **Define**: $G_{i,t} =r_{i,t} + \gamma r_{i,t+1} + \gamma^2 r_{i,t+1}+ \dots $ as the return from time step $t$ onwards in $i$th episode
- For state s visited at time step $t$ in episode $i$
  - Increment counter of total visits: $N(s) = N(s) + 1$
  - Update estimate
  $$V^{\pi}(s) = V^{\pi}(s)\frac{N(s)-1}{N(s)}+\frac{G_{i,t}}{N(s)} = V^{\pi}(s)+ \frac{1}{N(s)}\left(G_{i,t}-V^{\pi}(s) \right) $$
  - weight old estimate by $\frac{N(s)-1}{N(s)}$   and plus the new current state return $G_{i,t}$ by $\frac{1}{N(s)}$
  - $\left(G_{i,t}-V^{\pi}(s) \right)$ is kind of **learning rate** in machine learning
  

**Example**
- Sample episode $i = \{s_{i,1},a_{i,1},r_{i,1},s_{i,2},a_{i,2},r_{i,2},\dots, s_{i,T_i},a_{i,T_i},r_{i,T_i}\}$
- Define $G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^{T_i -t} r_{T_i} $ as return form time step $t$ onwards in $i$th episode
- for $t=1$: where $T_{i}$ is the length of the $i$th episode 
  - $$V^{\pi}(s_{i,t}) = V^{\pi}(s_{i,t})+ \alpha \left(G_{i,t}-V^{\pi}(s_{i,t}) \right) $$


![Expectation Tree](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%203/Expectation%20Tree.png)

### Evaluation of the Quality of a Policy Estimation Approach
> How good the policy is?


**Standard for Evaluation**
- Consistency: with enough data, does the estimate converge to the true value of the policy?
- Computational complexity: as get more data, computational cost of updating estimate
- Memory requirements 
- Statistical efficiency (intuitively, how does the accuracy of the estimate change with the amount of data)
- Empirical accuracy, often evaluated by mean squared error

**Evaluation of the Quality of a Policy Estimation Approach: Bias, Variance and MSE**

- Consider a statistical model that is parameterized by $\theta$ and that determines a probability distribution over observed data $P(x|\theta)$
- Consider a statistic $\hat{\theta}$ that provides an estimate of $\theta$ and is a function of observed data x
  - E.g. for a Gaussian distribution with known variance, the average of a set of i.i.d data points is an estimate of the mean of the Gaussian
- Definition: the bias of an estimator $\hat{\theta}$ is:
$$Bias_{\theta}(\hat{\theta}) = \mathbb{E}_{x|\theta}[\hat{\theta}]-\theta$$
- Definition: the variance of an estimator $\hat{\theta}$ is:
$$Var(\hat{\theta}) = \mathbb{E}_{x|\theta} [(\theta - \mathbb{E}[\hat{\theta}])^2]$$

- Definition: mean squared error (MSE) of an estimator $\hat{\theta}$ is:
$$MSE(\hat{\theta}) =Var(\hat{\theta}) +Bias_{\theta}(\hat{\theta})^{2}  $$


### Evaluation of the Quality of a Policy Estimation Approach: Consistent Estimator
- Let $n$ be the number of data points $x$ used to estimate the parameter $\theta$ and call the resulting estimate of $\theta$ using that data $\hat{\theta}$
- Then the estimator $\hat{\theta}$ is consistent if, for all $\varepsilon > 0$
$$\lim_{n \rightarrow \infty} Pr(|\hat{\theta}_{n} - \theta|> \varepsilon) > 0$$
- If an estimator is unbiased (bias = 0) is it consistent?


### Properties of Monte Carlo On Policy Evaluators
> Properties:
> - First-visit Monte Carlo
>   - $V^{\pi}$ estimator is an unbiased estimator of true $\mathbb{E}_{\pi}[G_t|s_t=s]$
>   - By law of large numbers, as $N(s)\rightarrow \infty$, $V^{\pi} \rightarrow \mathbb{E}_{\pi}[G_t|s_t=s]$
> - Every-visit Monte Carlo
>   - $V^{\pi}$ every-visit MC estimator is a **biased** estimator of $V^{\pi}$
>   - But consistent estimator and often has better MSE
> - Incremental Monte Carlo
>   - Properties depends on the learning rate $\alpha$
>   - Update is $V^{\pi}(s_{i,t}) = V^{\pi}(s_{i,t})+ \alpha_{k}(s_j) \left(G_{i,t}-V^{\pi}(s_{i,t}) \right)$
>   - where we have allowed $\alpha$ to vary (let $k$ be the total number of updates done so far, for state $s_{i,t} = s_j$ )
>   - incremental MC estimate will converge to true policy value $V^{\pi}(s_j)$ under condition: $$\begin{aligned}
    \sum_{n=1}^{\infty}\alpha_{n}(s_j) & = \infty \\
    \sum_{n=1}^{\infty}\alpha^{2}_{n}(s_j) & <\infty
\end{aligned}$$


### Monte Carlo (MC) Policy Evaluation Key Limitations
- Generally high variance estimator
  - Reducing variance can require a lot of data
  - In cases where data is very hard or expensive to acquire, or the stakes are high, MC may be impractical
- Requires episodic settings
  - Episode must end before data from episode can be used to update $V$


### Monte Carlo (MC) Policy Evaluation Summary
- **Aim**: estimate $V^{\pi}(s)$ given episodes generated under policy $\pi$
  - $\{s1, a1, r1, s2, a2, r2, . . . \}$where the actions are sampled from $\pi$
- **Accumulated Reward**:$G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^{T_i -t} r_{T_i} $ under policy $\pi$
- **State Value Function**: $V^{\pi}(s) =\mathbb{E}_{\pi}[G_t|s_t = s] $
- **Sample**: Estimates expectation by empirical average (given episodes sampled from policy of interest)
- **Update**: updates $V$ estimate using **sample** of return to approximate the expectation
- **Constriant**:Does **not** assume Markov process
- **Convergence**: Converges to true value under some (generally mild) assumptions
- **Note**: Sometimes is preferred over dynamic programming for policy evaluation even if know the true dynamics model and reward


## Temporal Difference (TD)
**Temporal Difference Learning**
- “If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be temporal-difference (TD) learning.” – Sutton and Barto 2017
- Combination of Monte Carlo & dynamic programming methods
- Model-free
- Can be used in episodic or infinite-horizon non-episodic settings
- Immediately updates estimate of $V$ after each $(s, a, r , s')$ tuple


### Temporal Di↵erence Learning for Estimating $V$

- **Aim**: estimate $V^{\pi}(s)$ given episodes generated under policy $\pi$
- **average return**
$$G_t =r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+ \dots + \gamma^{T_i -t} r_{T_i}
$$ in MDP under policy $\pi$
- **State Value Function**: $V^{\pi}(s) =\mathbb{E}_{\pi}[G_t|s_t = s] $
- Recall Bellman operator (if know MDP models)
$$B^{\pi}V(s) = r(s,\pi(s)) + \gamma \sum_{s' \in S}P(s'|s,\pi(s)) V(s') $$
- In incremental every-visit MC, update estimate using 1 sample of return (for the current $i$th episode)
$$V^{\pi}(s) =  V^{\pi}(s)+ \alpha \left(G_{i,t}-V^{\pi}(s) \right) $$
- **Idea**: have an estimate of $V^{\pi}(s)$, use to estimate expected return
$$V^{\pi}(s) =  V^{\pi}(s)+ \alpha \left(r_t+\gamma V^{\pi}(s_{t+1})-V^{\pi}(s) \right) $$


### Temporal Difference [TD(0)] Learning
- **Aim**: estimate $V^{\pi}(s)$ given episodes generated under policy $\pi$
  - $\{s_{i,1},a_{i,1},r_{i,1},s_{i,2},a_{i,2},r_{i,2},\dots, \}$ where the actions are sampled from $\pi$
- TD(0) learning / 1-step TD learning: update estimate towards target
$$V^{\pi}(s_t) = V^{\pi}(s_t)+\alpha \left(\underbrace{r_t+\gamma V^{\pi}(s_{t+1})}_{\text{TD Target}}-V^{\pi}(s) \right) $$
- TD(0) error:
$$\delta_{t} = r_t+\gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_{t})$$
- Can immediately update value estimate after $(s, a, r , s')$ tuple
- Don’t need episodic 

### Temporal Difference [TD(0)] Learning Algorithms
- Input $\alpha$
- Initialize $V^{\pi}(s)=0, \forall s \in S$
- Loop
  - Sample **tuple**  $(s_t, a_t, r_t , s_{t+1})$
  - $\delta_{t} = r_t+\gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_{t})$

**Temporal Di↵erence (TD) Policy Evaluation**
![TD Expectation Tree](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%203/TD%20Expectation%20Tree.png)


### Summary: Temporal Difference Learning
- Combination of Monte Carlo & dynamic programming methods
- Model-free
- **Bootstraps and samples**
- Can be used in episodic or infinite-horizon non-episodic settings
- Immediately updates estimate of V after each $(s, a, r , s')$ tuple
- Biased estimator (early on will be influenced by initialization, and won’t be unibased estimator)
- Generally lower variance than Monte Carlo policy evaluation
- Consistent estimator if learning rate ↵ satisfies same conditions specified for incremental MC policy evaluation to converge
- Note: algorithm I introduced is TD(0). In general can have approaches that interpolate between TD(0) and Monte Carlo approach

## Certainty Equivalence $V^{\pi}$ MLE MDP Model Estimates
- Model-based option for policy evaluation without true models
- After each $(s_i , a_i , r_i , s_{i+1})$ tuple
  - Recompute maximum likelihood MDP model for $(s, a)$
  $$\begin{aligned}
    \hat{P}(s'|s,a) & = \frac{1}{N(s,a)}\sum_{k=1}^{i}\mathbb{1}(s_k=s,a_k=a,s_{k+1}=s')\\
    \hat{r}(s,a) &= \frac{1}{N(s,a)}\sum_{k=1}^{i}\mathbb{1}(s_k=s,a_k=a)r_{k}
  \end{aligned}$$
  - Compute $V^{\pi}$ using MLE MDP (using any dynamic programming method from lecture 2)
- Optional worked example at end of slides for Mars rover domain.
- **Computational Cost**: Updating MLE model and MDP planning at each update ($O(|S|^3)$ for analytic matrix solution, $O(|S|^2|A|)$ for iterative methods)
- Very data efficient and very computationally expensive
- Consistent (will converge to right estimate for Markov models)
- Can also easily be used for off-policy evaluation (which we will shortly define and discuss)

## Batch Policy Evaluation
**Batch MC and TD**
- Batch (offline) solution for finite dataset
  - Given set of $K$ episodes
  - Repeatedly sample an episode from K
  - Apply MC or TD(0) to sampled episode
- What do MC and TD(0) converge to?