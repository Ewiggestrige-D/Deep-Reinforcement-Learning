# Stanford CS234 Reinforcement Learning | Introduction to Reinforcement Learning | 2024 | Lecture 2
[Stanford CS234 Reinforcement Learning | 2024 | Lecture 2](https://www.youtube.com/watch?v=gHdsUUGcBC0)


> **Question for today**: Can we construct algorithms for computing decision policies so that we can guarantee with additional computation / iterations, we monotonically improve the decision policy? 
> Do all algorithms satisfy this property?

## Making good decisions given a **Markov decision process(MDP)**

> Given a model under such context means given a dynamic models $P(s_{t+1} =s'|s_t =s)$ that tells us how the world evolves when we make decisions and the reward function $R(s_t=s)=\mathbb{E}[r_t|s_t=s]$ tells how good decisions are

### Computing the Value of a Markov Reward Process
- Markov property provides structure
- MRP value function satisfies
  $$V(s) = \underbrace{R(s)}_{\text{Immediate reward}} + \underbrace{\gamma \sum_{s' \in S}P(s'|s)V(s')}_{\text{Discounted sum of future rewards}} 
  $$
- For finite state MRP, we can express V(s) using a matrix equation
$$\begin{pmatrix}
V(s_1) \\
\vdots \\
V(s_N)
\end{pmatrix} = 
\begin{pmatrix}
R(s_1) \\
\vdots \\
R(s_N)
\end{pmatrix} +
\gamma \begin{pmatrix}
    P(s_1|s_1) & P(s_2|s_1) & \dots & P(s_N|s_1) \\
    P(s_1|s_2) & P(s_2|s_2) & \dots & P(s_2|s_N) \\
    \vdots & \vdots & \ddots &\vdots \\
    P(s_1|s_N) & P(s_2|s_N) & \dots & P(s_N|s_N)
\end{pmatrix}
\begin{pmatrix}
V(s_1) \\
\vdots \\
V(s_N)
\end{pmatrix}
$$
where
$$\begin{aligned}
    V & = R+\gamma PV \\
    V - \gamma PV & = R\\
    (\mathbb{I} -\gamma P)V & =R \\
    V & =(\mathbb{I} -\gamma P)^{-1}R
  \end{aligned}
$$
- Solving directly requires taking a matrix inverse $\sim O(N^3)$
- Note that $(\mathbb{I} -\gamma P)$ is invertible

---
**“Note that $(\mathbb{I} - \gamma P)$ is invertible”**

这句话强调了一个**关键的数学性质**：

- 虽然不是所有矩阵都可逆，但在这个特定情况下，**$(\mathbb{I} - \gamma P)$ 一定是可逆的**，因此公式 $V = (\mathbb{I} - \gamma P)^{-1} R$ 是**良定义的（well-defined）**。
- **为什么可逆？**  
  原因在于：
  - $P$ 是一个**随机矩阵**（每一行是非负数且和为 1）；
  - 折扣因子 $\gamma \in [0, 1)$（通常严格小于 1）；
  - 因此，矩阵 $\gamma P$ 的**谱半径（spectral radius）小于 1**；
  - 根据线性代数中的**Neumann 级数理论**，若 $\rho(\gamma P) < 1$，则 $(\mathbb{I} - \gamma P)$ 可逆，且其逆可表示为：
    $$
    (\mathbb{I} - \gamma P)^{-1} = \sum_{k=0}^{\infty} (\gamma P)^k
    $$
    这恰好对应了**未来折扣回报的无穷展开**！

> ✅ 所以，这个可逆性保证了 MRP 的值函数**存在且唯一**，这是强化学习理论的重要基础。


#### 总结

| 语句 | 含义 |
|------|------|
| “$(\mathbb{I} - \gamma P)$ is invertible” | 保证了解的存在性和唯一性，理论上有闭式解 |

---


### Iterative Algorithm for Computing Value of a MRP
- Dynamic programming (to avoid compute the inverse of matrix *directly*) 
- Initialize $V_{0}(s) = 0$ for all $s$
- For $k = 1$ until convergence, $k$ labels the epoch
  - For all $s \in S$
  $$V_{k}(s) = R(s) +\gamma \sum_{s'\in S}P(s'|s)V_{k-1}(s')$$
- Computational complexity: $O(|S|^2)$ for each iteration ($|S| = N$)

## Markov Decision Process (MDP)
- Markov Decision Process is **Markov Reward Process + actions**
- Definition of MDP
  - $S$ is a (finite) set of states ($s \in S$)
  - $A$ is a (finite) set of **Action** ($a \in A$)
  - $P$ is dynamics/transition models for ***each action*** that specofies $P(s_{t+1} =s'|s_t =s,a_t=a)$
  - $R$ is a reward function $$R(s_t=s,a_t=a)=\mathbb{E}[r_t|s_t=s,a_t=a]$$
  - Discount factor $\gamma \in [0,1]$
- MDP is a tuple $(S,A,P,R,\gamma)$
- Once you define the state space/ action space/ reward function/ gamma factor, the MDP is determined

> Reward is sometimes defined as a function of the **current state**, or as a function of
the **(state, action, next state) tuple**. Most frequently in this class, we will assume reward
is a function of **state and action**



**Example: Mars Rover MDP**
- 2 deterministic actions
$$P(s'|s,a_1) = 
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
\end{pmatrix}, \quad
P(s'|s,a_2) = 
\begin{pmatrix}
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{pmatrix}
$$

### MDP Policies
- Policy specifies what **action to take in each state** 
  - Can be deterministic or stochastic
- For generality, consider as a **conditional distribution**
  - Given a state, specifies a distribution over actions
- Mathematics: $\pi(s|a) = P(a_t =a | s_t =s)$

**MDP + Policy**
- MDP + $\pi(a|s)$ = Markov Reward Process
- Precisely, it is the MRP(Markov Reward Precess) $(S, R^{\pi}, P^{\pi}, \gamma)$, where
$$\begin{aligned}
    R^{\pi}(s) & = \sum_{a \in A}\pi(a|s)R(s,a) \\
    P^{\pi}(s'|s) &= \sum_{a \in A}\pi(a|s)P(s'|s) \\
\end{aligned}
$$
- Implies we can use *same* techniques to evaluate the value of a policy for a MDP as we could to compute the value of a MRP, by defining a MRP with $R^{\pi}$ and $P^{\pi}$

## **MDP Policy Evaluation, Iterative Algorithm**
- Initialize $V_{0}(s) = 0$ for all $s$
- For $k =1$ until convergence
  - For all $s \in S$
  $$ V_{k}^{\pi}(s) = \sum_{a}\pi(a|s) \left[R(s,a) + \gamma \sum_{ss' \in S}p(s'|s,a)V_{k-1}^{\pi}(s') \right]
  $$
- This is a ***Bellman Backup*** for a particular policy $\pi$
- It look very similar to MRP, except it is based on *propbability* for each action, what would we get next?
- Note the if the policy is deterministic then the above update simplifies to 
$$V_{k}^{\pi}(s) = R(s,\pi(s)) + \gamma \sum_{ss' \in S}p(s'|s,\pi(s))V_{k-1}^{\pi}(s') 
$$
which means in given state $s$, the only choice action is $a$, not a **action distrubution**

### MDP Control
- Compute the optimal policy
$$\pi^{*} = arg \max_{\pi}V^{\pi}(S)
$$
- There exists **a unique optimal value function** 
- Optimal policy for a MDP in an infinite horizon problem (agent acts forever is
  - **deterministic**
  - **Stationary** (does not depend on time step)
  - Unique? Not necessarily, may have two policies with identical (optimal) values

**Policy Search**
> To find a methods and algorithms that have monotonic improvement capabilities, one of the options is a **Policy Search**

- One option is searching to compute best policy
- Number of deterministic policies is $|A|^{|S|}$
- Policy iteration is generally more **efficient** than enumeration

**MDP Policy Iteration (PI)**
> main idea is to alter between having a candidate decision that might be **optimal**. We are going to evaluate it and going to see if we can improve it. If we can, we do it and if we do not, we halt.

- Set $i = 0$
- Initialize $\pi_0(s)$ randomly for all states $s$
- while $i == 0$ or $||\pi_i - \pi_{i-1}||_{1} > 0$ ($L$-1 norm, measures if the policy changed for any state)
  - $V^{\pi_{i}} \leftarrow$ MDP $V$ functionpolicy evaluation of $\pi_{i}$
  - $\pi_{i+1} \leftarrow$ Policy **improvement**
  - $i =i+1$ (iteration)


## New Definition: State-Action Value Q
- State-action value of a **policy**
$$Q^{\pi}(s,a) = R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V^{\pi}(s')$$
- Take action $a$ (at given state $s$), then follow the policy $\pi$

**Policy Improvement**
- Compute state-action value of a policy $\pi_{i}$
  - For $s \in S$ and $a \in A$:
  $$
  Q^{\pi_{i}}(s,a) = R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V^{\pi_{i}}(s')
  $$
- Compute new policy $\pi_{i+1} $, for all $s \in S$
$$
\pi_{i+1}(s) = arg \max_{a} Q^{\pi_{i}}(s,a), \quad \forall s \in S
$$

**Delving Deeper Into Policy Improvement Step**
> **Definition of Q-function**: 
> $$ Q^{\pi_{i}}(s,a) = R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V^{\pi_{i}}(s')$$

To improve the policy , the main idea is

- Suppose we take $\pi_{i+1}(s)$ for one action, then follow $\pi_{i}$ forever
- Our expected sum of rewards is at least as good as if we had always followed $\pi_{i}$
- But new proposed policy is to always follow $\pi_{i+1}(s)$

$$\max_{a}Q^{\pi_{i}}(s,a) \geq R(s,\pi_{i}(s)) + \gamma \sum_{s' \in S}P(s'|s,\pi_{i}(s))V^{\pi_{i}}(s') = V^{\pi_{i}}(s)
$$
$$
π_{i+1}(s)= \max_{a} Q^{\pi_{i}}(s,a)
$$

**Proof: Monotonic Improvement in Policy**
- Definition
$$ V^{\pi_{1}} \geq  V^{\pi_{2}}: V^{\pi_{1}}(s) \geq V^{\pi_{2}}(s), \quad \forall s\in S
$$

- Proposition:$V^{\pi_{i+1}} \geq  V^{\pi_{i}}$ with strict inequality if $\pi_{i}$ is suboptimal, where $\pi_{i+1}$ is the new policy we get from policy improvement on $\pi_{i}$

>**Proof**:
>$$\begin{aligned}
>V^{\pi_{i}}(s) & \leq \max_{a} Q^{\pi_{i}}(s,a) \\
>& = \max_{a} \left( R(s,a) + \gamma \sum_{s' \in S}P(s'|>s,a)V^{\pi_{i}}(s') \right) \\
>& = R(s,\pi_{i+1}(s)) + \gamma \sum_{s' \in S}P(s'|s,>\pi_{i+1}(s))V^{\pi_{i}}(s') // \text{by definiton of}  >\pi_{i+1}\\ 
>& \leq  R(s,\pi_{i+1}(s)) + \gamma \sum_{s' \in S}P(s'|>s,\pi_{i+1}(s)) \left( \max_{a'} Q^{\pi_{i}}(s',a')>\right) \\
>& =  R(s,\pi_{i+1}(s)) + \gamma \sum_{s' \in S}P(s'|s,>\pi_{i+1}(s)) \\
>& \quad \left(R(s',\pi_{i+1}(s')) + \gamma \sum_{s'' >\in S}P(s''|s',\pi_{i+1}(s')) \right) \\
>& \vdots \\
>& = V^{\pi_{i+1}}(s) 
>\end{aligned}  
>$$

### MDP: Computing Optimal Policy and Optimal Value
- Policy iteration computes infinite horizon value of a policy and then improves that policy
- Value iteration is another technique
  - Idea: Maintain optimal value of starting in a state s if have a finite number of steps k left in the episode
  - Iterate to consider longer and longer episodes

**Bellman Equation and Bellman Backup Operators**
- Value function of a policy must satisfy the Bellman equation
$$
V^{\pi}(s) = R^{\pi}(s) + \gamma \sum_{s' \in S}P^{\pi}(s'|s)V^{\pi_{i}}(s')
$$
- Bellman backup operator
  - Applied to a **value function**
  - Returns a new **value function**
  - **Improves** the value if possible
  $$BV(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V^{\pi}(s') \right]$$
  - $BV$ yields a value function over all states s

**Value Iteration (VI)**
- Set $k =1$ 
- Initialize $V_{0}(s) = 0$ for all $s$
- Loop until convergence: 
(for example $|V_{k+1} -V_{k}|_{\infty} \leq \varepsilon$ which means $\max_{s} |V_{k+1}(s) -V_{k}(s)|$  )
  - For each $s \in S$
  $$V_{k+1}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') \right]$$
  - View as Bellman backup on value function
  $$V_{k+1}(s) = BV_{k}(s)$$
  $$\pi_{k+1}(s) = arg \max_{a} \left[ R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') \right]$$

**Policy Iteration as Bellman Operations**
- Bellman backup operator $B^{\pi}$ for a particular policy is defined as
$$B^{\pi}V(s) = R^{\pi}(s) + \gamma \sum_{s' \in S}P^{\pi}(s'|s)V^{\pi}(s')$$
- Policy evaluation amounts to computing the fixed point of $B^{\pi}$
- To do policy ***evaluation***, repeatedly apply operator until $V$ stops changing
$$V^{\pi} = B^{\pi}B^{\pi}B^{\pi} \dots B^{\pi}V$$
- To do policy ***improvement***
$$BV(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V^{\pi}(s') \right]$$


### Contraction Operator
- Let $O$ be an operator,and $|x|$ denote (any) norm of $x$
- If $|OV − OV'| \leq |V − V'|$, then $O$ is a **contraction operator**

**Will Value Iteration Converge?**
- Yes, if discount factor $\gamma < 1$, or end up in a terminal state with probability 1
- Bellman backup is a contraction if *discount factor $\gamma < 1$*
- If apply it to two different value functions, distance between value functions shrinks after applying Bellman equation to each

> **Proof**:
> Let $ ||V − V'|| =||V − V'||_{\infty} = \max_{s} |V_{k+1}(s) -V_{k}(s)| $  be the ***infinity norm***
> $$\begin{aligned}
> ||BV_{i} − BV_{j}|| &= \left\|\max_{a} \left(R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') \right) -\max_> {a'} \left(R(s,a') + \gamma \sum_{s' \in S}P(s'|s,a')V_{j}(s') \right)\right\| \\
> & = \max_{a}\left\|R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') - R(s,a) - \gamma \sum_{s' \in S}P(s'|s,> a)V_{j}(s')\right\| \\
> & =\max_{a} \gamma \cdot \left\| \sum_{s' \in S}P(s'|s,a)\left(V_{k}(s') -  V_{j}(s') \right) \right\| \\
> & \leq \max_{a} \gamma \left\|V_{k}(s') -  V_{j}(s') \right\|_{\infty} \underbrace{\sum_{s' \in S}P(s'|s,a)}_> {=1} \\
> & = \gamma \left\|V_{k}(s') -  V_{j}(s') \right\|_{\infty} 
> \end{aligned} 
> $$
> - Note: Even if all inequalities are equalities, this is still a **contraction** if $\gamma < 1$
> - Prove value iteration converges to a unique solution for discrete state and action spaces with $\gamma < 1$


### Value Iteration for Finite Horizon $H$
- $V_k$ =  optimal **value** if making $k$ more decisions
- $\pi_k$ =  optimal **policy** if making $k$ more decisions
  - Initialize $V_{0}(s) = 0$ for all states $s$
  - For $k = 1 : H$ where $H$ is now finite
    - For each state $s$
    $$V_{k+1}(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') \right]$$
    $$\pi_{k+1}(s) = arg \max_{a} \left[ R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') \right]$$


### Computing the Value of a (specific) Policy in a Finite Horizon

- Alternatively can estimate by simulation
  - Generate a large number of episodes
  - Average returns
  - Concentration inequalities bound how quickly average concentrates to expected value
  - Requires **no assumption** of Markov structure

**Finite Horizon Policies**
- Set $k =1$ 
- Initialize $V_{0}(s) = 0$ for all $s$
- Loop until $k == H$: 
   - For each state $s$
    $$V_{k+1}(s) = \max_{a} R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s') $$
    $$\pi_{k+1}(s) = arg \max_{a}  R(s,a) + \gamma \sum_{s' \in S}P(s'|s,a)V_{k}(s')$$
> Is optimal policy stationary (independent of time step) in finite horizon tasks?
> In general no.

## Value vs Policy Iteration
- Value iteration:
  - Compute optimal value for horizon = $k$
    - Note this can be used to compute optimal policy if horizon = $k$
  - Increment $k$
- Policy iteration
  - Compute infinite horizon value of a policy
  - Use to select another (better) policy
  - Closely related to a very popular method in RL: **policy gradient**

## RL Terminology: Models, Policies, Values
- **Model**: *Mathematical* models of dynamics and reward
- **Policy**: Function mapping *states* to *actions*
- **Value function**: future rewards from being in a state and/or action when following a particular policy

## What You Should Know
- Define MP, MRP, MDP, Bellman operator, contraction, model, Q-value, policy
- Be able to implement
  - Value Iteration
  - Policy Iteration
- Give pros and cons of different policy evaluation approaches
- Be able to prove contraction properties
- Limitations of presented approaches and Markov assumptions
  - Which policy evaluation methods require the Markov assumption?