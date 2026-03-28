# Stanford CS234 Reinforcement Learning | Introduction to Reinforcement Learning | 2024 | Lecture 4
[Stanford CS234 Reinforcement Learning | 2024 | Lecture 4](https://www.youtube.com/watch?v=b_wvosA70f8)

# Model Free Control and Function Approximation

## Recap MC, TD(0) and Certainty Equivalence Policy Evaluation
- Policy evaluation: Estimate $V^{\pi}(s)$ from executing $\pi_i$
- Trajectories $\tau : (s, a \sim \pi(s), r , s′, a′ \sim \pi(s′), . . .)$ or tuples $(s, a, r , s′)$
- MC: Given a full trajectory  $\tau  : V^{\pi}(s) \leftarrow  (1 − \alpha(s))V^{\pi}(s) + \alpha G_t (s)$
- TD(0): Given $(s,a,r,s’)$
$V^{\pi}(s) \leftarrow (1 − \alpha(s))V^{\pi}(s)+ \alpha(s)(r + \gamma V^{\pi}(s'))$
- Certainty equivalence: Given a tuple $(s,a,r,s’)$, update MLE dynamics model and reward model and then use policy evaluation methods to
compute $V^{\pi}(s)$  for all $s$

## Batch MC and TD
- TD and MC methods shown use data **once**, then discard
- Batch (Offline) solution for **finite** dataset
  - Given set of $K$ episodes
  - *Repeatedly* sample an episode from $K$
  - Apply MC or TD(0) to the sampled episode
- What do MC and TD(0) converge to?

**Batch MC and TD: Convergence**
- Monte Carlo in batch setting converges to min MSE (mean squared error)
  - Minimize loss with respect to *observed returns*
- TD(0) converges to DP policy $V^{\pi}$ for the MDP with the maximum likelihood model estimates
- Aka same as **dynamic programming with certainty equivalence**!
  - Maximum likelihood Markov decision process model
  $$\begin{aligned}
    \hat{P}(s'|s,a) & = \frac{1}{N(s,a)}\sum_{k=1}^{i}\mathbb{1}(s_k=s,a_k=a,s_{k+1}=s')\\
    \hat{r}(s,a) &= \frac{1}{N(s,a)}\sum_{k=1}^{i}\mathbb{1}(s_k=s,a_k=a)r_{k}
  \end{aligned}$$


**Some Important Properties to Evaluate Model-free Policy Evaluation Algorithms**

Property:
- Data efficiency & Computational efficiency
- In simple TD(0), use $(s, a, r , s′)$ *once* to update $V(s)$
  - O(1) operation per update
  - In an episode of length L, O(L)
- In MC have to wait till episode **finishes**, then also O(L)
- MC can be more data efficient than simple TD
- But TD exploits *Markov structure*
  - If in Markov domain, leveraging this is helpful
- Dynamic programming with certainty equivalence also uses Markov structure


## Summary: Policy Evaluation
Estimating the expected return of a particular policy if don’t have access to true MDP models. 

Examples: evaluating average purchases per session of new product recommendation system

- Monte Carlo policy evaluation
  - Policy evaluation when we don’t have a model of how the world works
    - Given **on-policy** samples
    - Given **off-policy** samples
- Temporal Difference (TD)
- Dynamic Programming with certainty equivalence
- *Understand what MC vs TD methods compute in batch evaluations
- Metrics / Qualities to evaluate and compare algorithms
  - Uses Markov assumption
  - Accuracy / MSE / bias / variance
  - Data efficiency
  - Computational efficiency



## Model-free Policy Iteration
- Initialize policy $\pi$
- Repeat:
  - Policy evaluation: compute $Q^{\pi}$
  - Policy improvement: update $\pi$ given $Q^{\pi}(s,a)$
- May need to modify policy evaluation:
  - If $\pi$ is deterministic, can’t compute $Q(s, a)$ for any $a \neq \pi(s)$ 
- How to interleave policy evaluation and improvement?
  - Policy improvement is now using an estimated $Q$

**Problem of Exploration**
> how do we quantify our **uncertainty** in our knowledge and then how do we propagate that uncertainty into the value of that uncertainty for downstream decision-making


![Action-Reward](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%204/Action-reward.jpeg)

- Goal: Learn to select actions to maximize total expected future reward
- Problem: Can’t learn about actions **without** trying them (*need to explore*)
- Problem: But if we try new actions, spending less time taking actions that **our past experience suggests will yield high reward**(need to **exploit** knowledge of domain to achieve high rewards)


**$\varepsilon$-greedy Policies**
- Simple idea to balance exploration and achieving rewards
- Let $|A|$ be the number of actions
- Then an $\varepsilon$-greedy policy w.r.t. a state-action value $Q(s, a)$ is $π(a|s) =$
  - $arg \max_{a} Q(s,q)$ with probability $1-\varepsilon +\frac{\varepsilon}{|A|}$ (with high probability you're going to take whatever action maximizes your q value in your current state)
  - $a' \neq arg \max_{a} Q(s,q)$  with probability  $\frac{\varepsilon}{|A|}$
- In words: select argmax action with probability $1 − \varepsilon$, else select action uniformly at random

**Policy Improvement with $\varepsilon$-greedy policies**
- Recall we proved that policy iteration using given dynamics and reward models, was guaranteed to monotonically improve
- That proof assumed policy improvement output a deterministic policy
- Same property holds for $\varepsilon$-greedy policies

> ***Theorem***:
> For any $\varepsilon$-greedy policy $\pi_i$ , the $\varepsilon$-greedy policy w.r.t. $Q^{\pi_i}$ , $\pi_{i+1}$ is a monotonic improvement $V^{i+1}\geq V^{i}$
> $$\begin{aligned}
    Q^{\pi_i}(s,\pi_{i+1}(s)) & = \sum_{a \in A}\pi_{i+1}(a|s) Q^{\pi_i}(s,a) \\
    & = (\frac{\varepsilon}{|A|}) \left[ \sum_{a \in A}  Q^{\pi_i}(s,a) \right] + (1-\varepsilon)\max_{a} Q^{\pi_i}(s,a) \\
    & = (\frac{\varepsilon}{|A|}) \left[ \sum_{a \in A}  Q^{\pi_i}(s,a) \right] + (1-\varepsilon)\max_{a} Q^{\pi_i}(s,a)\frac{1-\varepsilon}{1-\varepsilon} \\
    & = (\frac{\varepsilon}{|A|}) \left[ \sum_{a \in A}  Q^{\pi_i}(s,a) \right] + (1-\varepsilon)\max_{a} Q^{\pi_i}(s,a)\sum_{a \in A}\frac{\pi_{i}(a|s)-\frac{\varepsilon}{|A|}}{1-\varepsilon} \\
    & \geq  (\frac{\varepsilon}{|A|}) \left[ \sum_{a \in A}  Q^{\pi_i}(s,a) \right] + (1-\varepsilon)\sum_{a \in A}\frac{\pi_{i}(a|s)-\frac{\varepsilon}{|A|}}{1-\varepsilon}Q^{\pi_i}(s,a) \\
    & =  \sum_{a \in A}\pi_{i}(a|s)Q^{\pi_i}(s,a) \\
    & = V^{\pi_i}(s)
\end{aligned}$$
Note: In short, the new policy that extract through doing  $\varepsilon$-greedy policy improvement  which is still an $\varepsilon$-greedy policy is going to be better than your old $\varepsilon$-greedy policy.


## Recall Monte Carlo Policy Evaluation, Now for $Q$-function
> 1: Initialize $Q(s, a) = 0,N(s, a) = 0 \forall(s, a), k = 1$, Input $\varepsilon = 1, \pi$
2: **loop**
3:      Sample $k$-th episode $(sk,1, ak,1, rk,1, sk,2, . . . , sk,T )$ given $\pi$
3:      Compute $G_{k,t} = r_{k,t} + \gamma r_{k,t+1} + \gamma^2 r_{k,t+2} + \dots+  \gamma^{Ti_−1}r_k, \quad T_i \forall t$
4:      **for** $t = 1, . . . ,T$ **do**
5:          **if** First visit to $(s,a)$ in episode $k$ then
6:                  $N(s, a) = N(s, a) + 1$ Counts add
7:                  $Q(s_t , a_t ) = Q(s_t , a_t ) + \frac{1}{N(s,a)}(G_{k,t} − Q(s_t , a_t ))$ 
8:          **end if**
9:      **end for**
10:     $k = k + 1$ new episode
11: **end loop**


### Monte Carlo Online Control / On-Policy Improvement
> 1: Initialize $Q(s, a) = 0,N(s, a) = 0 \forall(s, a), k = 1$, Input $\varepsilon = 1 $
> 2: $\pi_k = \varepsilon$ greedy ($Q$)
> 3: **loop** episode
> 4:    sample $k$-th episode $\{s_{k,1},a_{k,1},r_{k,1},s_{k,2},a_{k,2},r_{k,2},\dots,s_{k,T},a_{k,T},r_{k,T} \}$ given $\pi_k$
> 4:    Compute $G_{k,t} = r_{k,t} + \gamma r_{k,t+1} + \gamma^2 r_{k,t+2} + \dots+  \gamma^{Ti_−1}$
> 5:      **for** $t = 1, . . . ,T$ **do**
> 6:          **if** First visit to $(s,a)$ in episode $k$ then
> 7:                  $N(s, a) = N(s, a) + 1$ Counts add
> 8:                  $Q(s_t , a_t ) = Q(s_t , a_t ) + \frac{1}{N(s,a)}(G_{k,t} − Q(s_t , a_t ))$ 
> 9:          **end if**
> 10:      **end for**
> 11:     $k = k + 1, \quad \varepsilon = \frac{1}{k}$ new episode
> 12:     $\pi_k = \varepsilon$ greedy ($Q$) // Policy Impreovement to step 4  "given $\pi_k$"
> 12:     for each state $s$, $\pi(s) = arg \max_{a}Q(s,a)$ with probability $1-\varepsilon$ else random
> 13:**end loop**

Question:
- Is $Q$ an estimate of $Q^{\pi_k}$?
- when might this precedure fail to compute the optimal $Q^*$?

Answer:
- $Q$ is **not** an estimate of $Q^{\pi_k}$, because it is averaging over policies that are changing every episode. We are decaying $\varepsilon$ which means we're making things more and more deterministic. But in addition to that, our $Q$ might be changing. It's the weird weighted average of all the previous data and all the policies you've done before.
- And it should not be clear yet that we will necessarily converge to $Q^*$. We are getting more and more deterministic over time, because we're reducing epsilon. Eventually, we're going to converge towards something deterministic

**Properties of MC control with $\varepsilon$-greedy policies**
- Computational complexity?
- Converge to optimal $Q^∗$ (Optimal $Q$) function?
- Empirical performance?

### Greedy in the Limit of Infinite Exploration (GLIE)
> ***Definition*** of GLIE:
> - All state-action pairs are visited an infinite number of times
>   $$\lim_{i \rightarrow \infty} N_{i}(s,a) \rightarrow \infty , \quad \forall(s,a)$$
> - Behavior policy (policy used to act in the world) converges to greedy policy
>   $\lim_{i \rightarrow \infty} \pi(a|s) \rightarrow arg \max_{a}Q(s,a)$ with porobability 1

A simple GLIE strategy is $\varepsilon$-greedy where $\varepsilon$ is reduced to 0 with the following rate: $\varepsilon_i = \frac{1}{i}$

**GLIE Monte-Carlo Control using Tabular Representations**
> ***Theorem***:
> GLIE Monte-Carlo control converges to the optimal state-action value function $Q(s,a) \rightarrow Q^{*}(s,a)$


## Model-free Policy Iteration with TD Methods
- Initialize policy $\pi$
- Repeat:
  - *Policy evaluation*: compute $Q^\pi$ using temporal difference updating with $\varepsilon$-greedy policy
  - *Policy improvement*: Same as Monte carlo policy improvement, set$\pi$ to $\varepsilon$-greedy ($Q^{\pi}$)
- Method 1：SARSA (short for tuple $\{s,a,r,s',a'\}$)
- On-policy: SARSA computes an estimate $Q$ of policy used to act


**On and Off-Policy Learning**
- On-policy learning
  - Direct experience
  - Learn to estimate and evaluate a policy from experience obtained from following **that policy**
- Off-policy learning
  - Learn to estimate and evaluate a policy using experience gathered from following a **different policy**

**Q-Learning: Learning the Optimal State-Action Value**
- $Q$-learning
  - estimate the $Q$ value of $\pi^*$ while acting with another behavior policy $\pi_b$
  - Key idea: Maintain $Q$ estimates and bootstrap for best future value
- $Q$-learning:
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha(r_t +\gamma \max_{r'}Q(s_{t+1},a')-Q(s_t,a_t))$$
- For comparison TD(0) learning / 1-step TD learning: update estimate towards target
$$V^{\pi}(s_t) = V^{\pi}(s_t)+\alpha \left(\underbrace{r_t+\gamma V^{\pi}(s_{t+1})}_{\text{TD Target}}-V^{\pi}(s) \right) $$
- So estimate of $Q^{*}(s,a) = r_t +\gamma \max_{r'}Q(s_{t+1},a')$

### $Q$-Learning with $\varepsilon$-greedy Exploration

> 1：Initialize $Q(s,a), \forall s\in S, a\in A, t=0$, initial state $s_t=s_0$
> 2: Set $\pi_b$ to be $\varepsilon$-greedy w.r.t $Q(s,a,r,s',a')$
> 3:**Loop**
> 4:    Take $a_t \sim \pi_b(s_t)$ //sample action from policy
> 5:    Observe $(r_t,s_{t+1})$
> 6:    $$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha(r_t +\gamma \max_{r'}Q(s_{t+1},a')-Q(s_t,a_t))$$
> 7:    $$\pi(s_t) = arg \max_{a}Q(s_t,a)$$ with probability $1-\varepsilon$, else random
> 8:    $t=t+1,\varepsilon = 1/t$
> 9: **EndL Loop**


**Convergence Properties of Q-Learning
Theorem**
> ***Theorem***:
> Q-Learning for finite-state and finite-action MDPs converges to the
optimal action-value, $Q(s, a) → Q^∗(s, a)$, under the following conditions:
> 1. The policy sequence $\pi_t (a|s)$ satisfies the condition of GLIE
> 2. The step-sizes $\alpha_t$ satisfy the Robbins-Munro sequence such that
$$\begin{aligned}
    \sum_{n=1}^{\infty}\alpha_{n}(s_j) & = \infty \\
    \sum_{n=1}^{\infty}\alpha^{2}_{n}(s_j) & <\infty
\end{aligned}$$

For example $\alpha_t =\frac{1}{T}$ satisfies the above condition

**Properties of TD-Style Tabular Control with $\varepsilon$-greedy policies**
- Result builds on stochastic approximation
- Relies on step sizes decreasing at the right rate
- Relies on Bellman backup contraction property
- Relies on bounded rewards and value function
- Note: other variants exist. SARSA (on-policy algorithm)

**Q-learningL Learning the optimal State-Action Value**
- SARSA is an **on-pulicy** learning algorithm
  - Estimates the value of behavior policy (policy using to take actions in the world)
  - and then updates the behabior policy
- Q-Learning
  - estimate the $Q$ value of $\pi^{*}$ while acting with another bahevior policy $\pi_b$
- Key Idea: Maitain $Q$ estimates and bootstrap for best future value 
- Comparison
  - SRASA:
   $$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha(r_t +\gamma \max_{r'}Q(s_{t+1},a_{t+1})-Q(s_t,a_t))$$
  - Q-Learning:
   $$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha(r_t +\gamma \max_{r'}Q(s_{t+1},a')-Q(s_t,a_t))$$


## Motivation for Function Approximation
- Avoid explicitly storing or learning the following for every single state and action
  - Dynamics or reward model $(P,R)$
  - Value $V$
  - State-action value $Q$
  - Policy $\pi$
- Want more compact representation that generalizes across state or states and actions
  - Reduce memory needed to store $(P, R)/V/Q/π$
  - Reduce computation needed to compute $(P, R)/V/Q/π$
  - Reduce experience needed to find a good $(P, R)/V/Q/π$


**State Action Value Function Approximation for Policy Evaluation with an Oracle**
- First assume we could query any state s and action a and an oracle would return the true value for $Q^{\pi}(s, a)$
- Similar to supervised learning: assume given $((s, a),Q^{\pi}(s, a))$ pairs
- The objective is to find the best approximate representation of $Q^{\pi}(s, a)$ given a particular parameterized function $\hat{Q}(s, a;w)$


### **Stochastic Gradient Descent**
- Goal:Find the parameter vector $\mathbf{w}$ that minimizes the loss betwwen a true value function $Q^{\pi}(s,a)$ and its approximation $\hat{Q}(s,a;\mathbf{w})$ as reprensted with a particular function class parameterized by $\mathbf{w}$
- Generalluy use mean squared error and define the loss as 
$$J(\mathbf{w})=\mathbb{E}_{\pi}[(Q^{\pi}(s,a)-\hat{Q}(s,a;\mathbf{w}))^2]$$
- Can use **Gradient descent** to find a local minimum
$$\delta \mathbf{w} = -\frac{1}{2}\alpha\nabla_{\mathbf{w}}J(\mathbf{w})$$
- Stochastic graiednt descent (SGD) uses a finite bumber of (often one) sample to copute an approximate gradient:
$$\begin{aligned}
    \nabla_{\mathbf{w}}J(\mathbf{w}) &= \nabla_{\mathbf{w}} \mathbb{E}_{\pi}[(Q^{\pi}(s,a)-\hat{Q}(s,a;\mathbf{w}))^2] \\
    & = -2\mathbb{E}_{\pi}[Q^{\pi}(s,a)-\hat{Q}(s,a;\mathbf{w})]\nabla_{\mathbf{w}}\hat{Q}(s,a;\mathbf{w})
\end{aligned}$$
- Expected SGD is the same as the full gradient update

## Model Free VFA Policy Evaluation
- No oracle to tell true $Q^{\pi}(s, a)$ for any state $s$ and action $a$
- Use model-free state-action value function approximation

### Model Free VFA Prediction / Policy Evaluation
- Recall model-free policy evaluation (Lecture 3)
  - Following a fixed policy $\pi$ (or had access to prior data)
  - Goal is to estimate $V^{\pi}$ and/or $Q^{\pi}$
- Maintained a lookup table to store estimates $V^{\pi}$ and/or $Q^{\pi}$
- Updated these estimates after each episode (Monte Carlo methods) or after each step (TD methods)
- Now: in value function approximation, change the estimate update step to include fitting the function approximator

### Monte Carlo Value Function Approximation
- Return $G_t$ is an **unbiased** but **noisy** sample of the true expected return $Q^{\pi}(s_t , a_t )$
- Therefore can reduce MC VFA to doing supervised learning on a set of (state,action,return) pairs:
$⟨(s_1, a_1),G_1⟩, ⟨(s_2, a_2),G_2⟩, . . . , ⟨(s_T , a_T ), G_T ⟩$
  - Substitute $G_t$ for the true $Q^{\pi}(s_t , a_t )$ when fit function approximator

> 1: Initialize $\mathbf{w},k =1$
> 2: **Loop**
> 3:    sample $k$-th episode $\{s_{k,1},a_{k,1},r_{k,1},s_{k,2},a_{k,2},r_{k,2},\dots,s_{k,T},a_{k,T},r_{k,T} \}$ given $\pi$
> 4:    **for** $t=1,\dots,L_k$ **do**
> 5:        **if** First visit to $(s,a)$ in episode $k$ **then**
> 6:        $$G_t(s,a) = \sum_{j=t}^{L_k}r_{k,j}$$
> 7:        $$\nabla_{\mathbf{w}}J(w) = -2[G_t(s,a)-\hat{Q}(s,a;\mathbf{w})]\nabla_{\mathbf{w}}\hat{Q}(s,a;\mathbf{w})$$ (Compute **Gradient**)
> 8:        Update wight $\delta \mathbf{w}$
> 9:        **end if**
> 10:    **end for**
> 11:   $k=k+1$
> 12:**End Loop**

### Temporal Difference TD(0) Policy Evaluation

**Recall: Temporal Difference Learning with Lookup Table**
- Uses bootstrapping and sampling to approximate $V^{\pi}$
- Updates $V^{\pi}(s)$ after each transition $(s, a, r , s′)$:
$$V^{\pi}(s) = V^{\pi}(s) + \alpha(r + \gamma V^{\pi}(s′) − V^{\pi}(s))$$
- Target is $r + \gamma V^{\pi}(s′)$, a **biased** estimate of the true value $V^{\pi}(s)$
- Represent value for each state with a separate table entry
- 3 forms of approximation:
   1. Sampling
   2. Bootstrapping
   3. Value function approximation

**Temporal Difference TD(0) Learning with Value Function Approximation**
- In value function approximation, target is $r + \gamma V^{\pi}(s′)$, a **biased** estimate of the true value $V^{\pi}(s)$
- Can reduce doing TD(0) learning with value function approximation to supervised learning on a set of data pairs:
$$<(s_1, r_1+\gamma \hat{V}^{\pi}(s_2;\mathbf{w}))>,<(s_2, r_2+\gamma \hat{V}^{\pi}(s_3;\mathbf{w}))>,\dots$$
- Find weights to minimize mean squared error
$$J(\mathbf{w}) = \mathbb{E}^{\pi}[(r_j + \gamma \hat{V}^{\pi}(s_{j+1};\mathbf{w})) - \hat{V}^{\pi}(s_j;\mathbf{w})^2]$$
- Use stochastic gradient descent, as in MC methods
> 1: Initialize $\mathbf{w},s$
> 2: **Loop**
> 3:    Given $s$ sample $a \sim \pi(s),r(s,a),s'\sim p(s'|s,a)$
> 4:    **for** $t=1,\dots,L_k$ **do**
> 5:         $$\nabla_{\mathbf{w}}J(w) = -2[r_j + \gamma \hat{V}^{\pi}(s_{j+1};\mathbf{w})) - \hat{V}^{\pi}(s_j;\mathbf{w}]\nabla_{\mathbf{w}}\hat{V}(s;\mathbf{w})$$ (Compute **Gradient**)
> 6:        Update wight $\delta \mathbf{w}$
> 7:        **if** $s'$ is not a terminal state **then**
> 8:        Set $s=s'$
> 9:        **else**
> 10:       Restart episode, sample initial state $s$
> 11:        **end if**
> 12:**End Loop**


## Control using Value Function Approximation
- Use value function approximation to represent state-action values $\hat{Q}^{\pi}(s,a;\mathbf{w})\approx Q^{\pi}$
- Interleave
  - Approximate policy evaluation using value function approximation
  - Perform $\varepsilon$-greedy policy improvement
- Can be *unstable*. Generally involves intersection of the following:
  - Function approximation
  - Bootstrapping
  - Off-policy learning

### Action-Value Function Approximation with an Oracle
- $\hat{Q}^{\pi}(s,a;\mathbf{w})\approx Q^{\pi}$
- Minimize the mean-squared error between the true action-value function $Q^{\pi}(s,a)$ and the approximate action-value function:
$$J(\mathbf{w})=\mathbb{E}_{\pi}[(Q^{\pi}(s,a)-\hat{Q}(s,a;\mathbf{w}))^2]$$
- Use **stochastic gradient descent** to find a local minimum
$$\nabla_{\mathbf{w}}J(\mathbf{w})= -2\mathbb{E}_{\pi}[Q^{\pi}(s,a)-\hat{Q}^{\pi}(s,a;\mathbf{w})]\nabla_{\mathbf{w}}\hat{Q}(s,a;\mathbf{w})
$$
- Stochastic gradient descent (SGD) samples the gradient

### Incremental Model-Free Control Approaches
- Similar to policy evaluation, **true state-action value function** for a state is unknown and so substitute a target value for true $Q(s_t , a_t )$
$$\Delta \mathbf{w} = \alpha (Q(s_t,a_t)-\hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$$
- In **Monte Carlo** methods, use a return $G_t$ as a substitute target
$$\Delta \mathbf{w} = \alpha (G_t- \hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$$
- **SARSA**: Use TD target $r + \gamma \hat{Q}(s', a';\mathbf{w})$ which leverages the current function approximation value
$$\Delta \mathbf{w} = \alpha (r + \gamma \hat{Q}(s', a';\mathbf{w})- \hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$$
- **Q-learning**: Uses related TD target $r + \gamma \max_{a'} \hat{Q}(s',a';w)$
$$\Delta \mathbf{w} = \alpha (r + \gamma \max_{a'} \hat{Q}(s', a';\mathbf{w})- \hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$$


### ”Deadly Triad” which Can Cause Instability
- Informally, updates involve doing an (approximate) *Bellman backup* followed by best trying to fit underlying value function to a particular feature representation
- Bellman operators are contractions, but value function approximation fitting can be an expansion
  - To learn more, see Baird example in Sutton and Barto 2018
- ”Deadly Triad” can lead to oscillations or lack of convergence
- Bootstrapping
- Function Approximation
- Off policy learning (e.g. Q-learning)


### Deep Q-Learning
- Q-learning converges to optimal $Q^∗(s, a)$ using tabular representation
- In value function approximation Q-learning minimizes MSE loss by stochastic gradient descent using a target Q estimate instead of true Q
- But Q-learning with VFA can diverge
- Two of the issues causing problems:
  - Correlations between samples
  - Non-stationary targets
- Deep Q-learning (DQN) addresses these challenges by using
  - Experience replay(Buffer Replay)
  - Fixed Q-targets (Fixed DQN)


### **DQNs: Experience Replay**
- To help remove correlations, store dataset $\mathcal{D}$ from prior experience
![Buffer Replayu](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%204/buffer%20replay.jpeg)

- To perform experience replay, repeat the following:
  - $(s, a, r , s′) \sim \mathcal{D}$: sample an experience tuple from the dataset
  - Compute the target value for the sampled $s$: $r + \gamma \max_{a'} \hat{Q}(s',a';w)$
  - Use stochastic gradient descent to update the network weights
  $$\Delta \mathbf{w} = \alpha (r + \gamma \max_{a'} \hat{Q}(s', a';\mathbf{w})- \hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$$
- **Uses target as a scalar, but function weights will get updated on the next round, changing the target value**

### DQNs: Fixed Q-Targets
- To help improve stability, fix the *target weights* used in the target calculation for multiple updates
- *Target network uses a different set of weights than the weights being updated*
- Let parameters $\mathbf{w}^{-}$ be the set of weights used in the **target**, and $\mathbf{w}$ be the weights that are being **updated**
- Slight change to computation of target value:
  - $(s, a, r , s′) \sim \mathcal{D}$: sample an experience tuple from the dataset
  - Compute the target value for the sampled $s$: $r + \gamma \max_{a'} \hat{Q}(s',a';w)$
  - Use stochastic gradient descent to update the network weights
  $$\Delta \mathbf{w} = \alpha (r + \gamma \max_{a'} \hat{Q}(s', a';\mathbf{w})- \hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$$


### DQN Pseudocode
> 1: Input $C, \alpha, \mathcal{D}=\{\}$,Initialize $\mathbf{w},\mathbf{w}^{-}=\mathbf{w},t=0$
> 2: Get initial state $s$
> 3: **Loop**
> 4:    Sample action $a_t$ given $\varepsilon$-greedy policy for current $\hat{Q}(s_t,a;\mathbf{w})$
> 5:    Obserce reward $r_t$ and next state $s_{t+1}$
> 6:    Store transition $(s_t,a_t,s_{t+1},a_{t+1})$ in replay Buffer $\mathcal{D}$
> 7:    Sample random minibatch of tuples $(s_i,a_i,s_{i+1},a_{i+1})$ from $\mathcal{D}$
> 8:    **for** $j$ in minibatch **do**
> 9:        **if** episode terminated at step $i + 1$ **then**
> 10:        $y_i=r_i$
> 11:       **else**
> 12:       $y_i=r_i+\gamma \max_{a'} \hat{Q}(s',a';w)$
> 13:       **end if**
> 14:       Do gradient descent step on $(y_i-\hat{Q}(s_i,a_i'\mathbf{w}))^2$ for parameters $\mathbf{w}:\Delta \mathbf{w} = \alpha (y_i-\hat{Q}(s_t,a_t;\mathbf{w}))\nabla_{\mathbf{w}}\hat{Q}(s_t,a_t;\mathbf{w})$
> 15:   **end for**
> 16:   $t=t+1$
> 17:   if $\mod(t,C) ==1$
> 18:       $\mathbf{w}^{1}\leftarrow \mathbf{w} $
> 19:   **end if**
> 20: **end loop**

Note:  there are several **hyperparameters** and **algorithm** choices. One needs to choose the **neural network architecture**, the **learning rate**, and **how often** to update the target network. Often a fixed size replay buffer is used for experience replay, which introduces a parameter to control the size, and the need to decide how to populate it.
### DQNs Summary
- DQN uses experience replay and fixed Q-targets
- Store transition $(s_t , a_t , r_{t+1}, s_{t+1})$ in replay memory $\mathcal{D}$
- Sample random mini-batch of transitions  $(s, a, r , s′)$ from $\mathcal{D}$
- Compute Q-learning targets w.r.t. old, fixed parameters  $\mathbf{w}^{-}$
- Optimizes MSE between Q-network and Q-learning targets
- Uses stochastic gradient descent


### Which Aspects of DQN were Important for Success?
![DQN with Tricks](/Stanford%20CS234%20I%20Reinforcement%20Learning%20I%20Spring%202024%20I%20Emma%20Brunskill/img/Stanford%20CS234%20%202024%20%20Lecture%204/DQN%20With%20tricks.png)

- Success in Atari has led to huge excitement in using **deep neural networks** to do value function approximation in RL
- Some immediate improvements (many others!)
  - Double DQN (Deep Reinforcement Learning with Double Q-Learning, Van Hasselt et al, AAAI 2016)
  - Prioritized Replay (Prioritized Experience Replay, Schaul et al, ICLR 2016)
  - Dueling DQN (best paper ICML 2016) (Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, ICML 2016)