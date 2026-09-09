---
title: LLM Learning Dynamics follows Evolutionary Game Theory
tldr: I came across an ODE system that describes the learning dynamics of classification-based models (like LLMs) under eNTK regime, which coincides with the well-studied replicator dynamics in game theory.
author: Jiatong Yu
date: May 2025
description: Game Theory blog with MathJax
---

## Motivation

I was reading [a paper](https://arxiv.org/abs/2407.10490) on LLM learning dynamics, and realized that their results can be extended into continuous time without any additional assumptions. Then I found the resulting ODE system that describes LLM learning dynamics coincides with the well-studied replicator dynamics in evolutionary game theory.

There are many low-hanging fruits to explore this direction, especially since the results from replicator dynamics is directly transferable. For example, [this paper](https://arxiv.org/abs/1507.07823) describes the attractors of the families of ODEs of our interest—here attractors can be interpreted as the convergence points of LLM learning dynamics.

## Preliminaries

::: item LLM Learning Dynamics
Let $V := |\mathcal{Y}|$ be the number of labels. A classification model $\theta$ processes input $x \in \R^d$ and produce output logits $z := f(x, \theta) \in \R^{V}$. We use $y \in \R^{V}$ to denote the label vector and $y_i \in \mathcal{Y}$ a particular label. Let $\pi_\theta(y_i|x)$ be the probability mass assigned to label $y_i$ by model state $\theta$, obtained through $\pi_\theta(y|x) = \softmax(z)$. For a training data point $(x_u,y_u)$, we focus on cross-entropy loss given by $\loss(\theta,x_u,y_u) = - \log \pi_\theta(y_u|x_u)$.

Following the empirical neural tangent kernel (eNTK) framework, we roll out the gradient of cross-entropy loss $\nabla_{\theta} \log \pi_{\theta_{t}}(y_u|x)$ into an *output gradient* term and a *network gradient* term:
\begin{equation*}
\begin{aligned}
& \nabla_{\theta_{t}} \log \pi_{\theta_{t}} (y_i | x) = \underbrace{\nabla_{z}\log \pi_{\theta_{t}}(y_i | \mathbf{x})^\top }_{\text{Output Gradient}} \underbrace{\nabla_{\theta_{t}} f(x,\theta_t)}_{\text{Network Gradient}}
\end{aligned}
\end{equation*}

Let $x_o$ be such test-time input. We want to trace the change in $\pi_\theta(y_i|x_o)$ for all labels $y_i\in \mathcal{Y}$. Formally, one-step learning dynamics in this work is defined as:
\begin{equation*}
\begin{aligned}
& \Delta \log \pi_{\theta_{t}}(y | x_o) \delequal \log \pi_{\theta_{t+1}} (y | x_o) - \log \pi_{\theta_{t}}(y | x_o)
\end{aligned}
\end{equation*}

Let $J(x_u,\theta_t) := \nabla_{\theta} f(x_u,\theta_t) \in \R^{V\times d}$ be the Jacobian matrix $J_{ij} = \frac{\partial f_i(x_u,\theta_t)}{\partial \theta_t^{(j)}}$. Define matrix-valued function $K(\cdot) : \R^{d} \times \R^d \rightarrow \R^{V \times V}$:
\[K_t(x_o,x_u) \delequal J_t(x_o) J_t(x_u)^\top\]
Let $e_i \in \R^d$ be the standard basis vector for label $y_i$, which has $0$ for all entries except the $y_i$-th entry. Define probability vector $\pi_t(x) := \pi_{\theta_{t}} (y|x) \in \R^{V}$ for simplicity. Apply a simple first-order Taylor expansion around $\pi_{\theta_{t+1}}$ to obtain the approximated single-step learning dynamics:
\begin{equation}\label{eq: single-step-update}
\begin{aligned}
\Delta\log \pi_{\theta_{t}}(y_i|x_o) & \approx \eta \big[ e_i - \pi_t(x_o) \big]^\top K_t(x_o,x_u) \big[ e_u - \pi_t(x_u)\big]
\end{aligned}\tag{1}
\end{equation}
We'll come back to this equation later.
:::

::: item Replicator Dynamics
Now let's switch gears to evolutionary game theory. For a more detailed introduction, see [this excellent post](https://stefanoallesina.github.io/Theoretical_Community_Ecology/game-theory-and-replicator-dynamics.html).

Imagine a large population of individuals, where each individual can adopt one of $m$ different strategies. Let $x_k(t)$ be the proportion of the population using strategy $k$ at time $t$. The vector $x(t) = (x_1(t), \dots, x_m(t))$ represents the state of the population.

The success of a strategy is determined by its **payoff** or **fitness**. In a simple symmetric game, we can define a payoff matrix $A$, where $A_{ij}$ is the payoff for an individual using strategy $i$ when interacting with an individual using strategy $j$. The expected fitness of an individual using strategy $k$ in a population with state $x$ is $f_k(x) = (Ax)_k$. The average fitness of the entire population is $\bar{f}(x) = x^\top A x$.

**Replicator dynamics** provides a simple, intuitive model for how the population state $x(t)$ evolves. The core idea is that strategies with above-average fitness will become more common, while those with below-average fitness will decline. This is captured by the replicator differential equation:
\begin{equation}\label{app eq: replicator ODE}
\dot{x}_k = x_k \Big( \underbrace{f_k(x) - \bar{f}(x)}_{\text{Fitness Advantage}} \Big)
\end{equation}

The rate of change of strategy $k$'s proportion, $\dot{x}_k$, is proportional to its current proportion $x_k$ and its "fitness advantage"—the difference between its own fitness and the population average. If a strategy's fitness is higher than the average, its share of the population grows. If it's lower, its share shrinks. This simple rule can produce rich and complex dynamics.
:::

## Continuous-Time Learning Dynamics of Language Models

The key to connecting these two worlds is the **lazy training regime**. In this regime, the model's internal representation of features change very little during training, which implies that the NTK matrix $K_t(x_o, x_u)$, remains nearly constant. We can approximate it as being fixed: $K_t(x_o,x_u) \approx K(x_o,x_u)$.

This "fixed feature" assumption allows us to transition from discrete, step-by-step updates to a continuous-time differential equation, making it possible to analyze the long-term behavior of learning. To see how the model's prediction at a test point $x_o$ evolves, we must simultaneously track the evolution of its predictions on all training points $\{x_u\}$, as they all influence each other through the fixed kernel matrix $K$.

::: item Warmup: Single Data Point Training
Let's start with the simplest case: training repeatedly on a single data point $(x_u, y_u)$. Using our learning dynamics equation (1), we can derive a discrete update rule for the model's predicted probabilities. After each step, we re-normalize the probabilities to ensure they sum to one. This gives the following system, which describes how predictions at the test point $x_o$ and training point $x_u$ co-evolve. Note that our framework handles both gradient descent ($\eta > 0$) and gradient ascent ($\eta < 0$).
\begin{equation}\label{eq: discrete-update-single}
\begin{aligned}
\tilde{\pi}_{t+1}(y_i|x_o)
&= \pi_{t}(y_i|x_o)\exp\!\Big\{\eta\,[e_i-\pi_t(x_o)]^\top K(o,u)[e_u-\pi_t(x_u)]\Big\},\\
\pi_{t+1}(y_i|x_o) &\propto \tilde{\pi}_{t+1}(y_i|x_o),\\[4pt]
\tilde{\pi}_{t+1}(y_i|x_u)
&= \pi_{t}(y_i|x_u)\exp\!\Big\{\eta\,[e_i-\pi_t(x_u)]^\top K(u,u)[e_u-\pi_t(x_u)]\Big\},\\
\pi_{t+1}(y_i|x_u) &\propto \tilde{\pi}_{t+1}(y_i|x_u).
\end{aligned}\tag{2}
\end{equation}

The following box says that the Eq. (2) is a well-behaved discretization of some ODE system. As the learning rate $\eta$ gets smaller, the trajectory of the discrete updates gets closer and closer to the trajectory of the ODE.

::: theorem Continuous-Time Learning Dynamics of Single Data Point Training
Let $V \ge 2$ and fix matrices $K,S \in \R^{V\times V}$. Let $u \in \{1,\dots,V\}$ and write $e_u \in \R^V$ for the $u$-th standard basis vector. Set
\[
\Delta^{V-1}:=\{x\in \R^V_{\ge 0}\mid \mathbf{1}^\top x=1\},\quad \mathbf{1}=(1,\dots,1)^\top.
\]

*ODE system.* For $(a,b)\in \R^V\times \R^V$ let
\begin{equation*}
\begin{aligned}
& F(a,b) = \begin{pmatrix}f(a,b)\\ g(a,b)\end{pmatrix}, \\
& f_i(a,b) = a_i\Big[(K(e_u-b))_i-a^\top K(e_u-b)\Big], \\
& g_i(a,b) = b_i\Big[(S(e_u-b))_i-b^\top S(e_u-b)\Big].
\end{aligned}
\end{equation*}
Let $z(t)=(a(t),b(t))$ be the solution of $\dot z=F(z)$ with initial condition $z(0)\in \Delta^{V-1}\times \Delta^{V-1}$.

*Discrete scheme.* Define the discrete update with initial value $z_0:=(a_0,b_0)=z(0)$. For a step size $\eta>0$ define $z_n=(a_n,b_n)$ recursively by
\[
\tilde a_{n+1,i}=a_{n,i}\exp\Big\{\eta\big[(K(e_u-b_n))_i-a_n^\top K(e_u-b_n)\big]\Big\},\quad
a_{n+1,i}=\frac{\tilde a_{n+1,i}}{\sum_j \tilde a_{n+1,j}},
\]
\[
\tilde b_{n+1,i}=b_{n,i}\exp\Big\{\eta\big[(S(e_u-b_n))_i-b_n^\top S(e_u-b_n)\big]\Big\},\quad
b_{n+1,i}=\frac{\tilde b_{n+1,i}}{\sum_j \tilde b_{n+1,j}}.
\qquad\text{(MU)}
\]

<p class="theorem-subtle">For every $T>0$ there exists $C=C\!\left(T,z_0,\|K\|_2,\|S\|_2\right)>0$ such that, for all $n\in\mathbb{N}$ with $t_n:=n\eta\le T$, the solution remains in the probability simplex and $\|z(t_n)-z_n\|\le C|\eta|$.</p>
:::
:::

::: item Full-Batch Training with Multiple Epochs
While training on a single data point is instructive, a more realistic setup involves a full dataset $D = \{(x_u, y_u)\}_{u=1}^N$. With full-batch gradient descent, the model's parameters are updated based on the average gradient across all training examples:
\[
\theta_{t+1} \leftarrow \theta_{t} + \eta \sum_{u=1}^N \nabla_\theta \log \pi_\theta(y_u | x_u) \Big|_{\theta = \theta_t}
\]
The change in log-probabilities at our test point $x_o$ is now an average of the influences from all training points:
\[
\Delta\log \pi_{\theta_t}(y_i|x_o)\approx
\frac{\eta}{N}\sum_{u=1}^N\big[e_i-\pi_t(x_o)\big]^\top K(o,u)\big[e_u-\pi_t(x_u)\big].
\]

This leads to a more complex, but analogous, discrete update rule for the probabilities:
\begin{equation}\label{eq: multi-update}
\tilde\pi_{t+1}(y_i|x_o)=\pi_t(y_i|x_o)\exp\!\left\{\frac{\eta}{N}\sum_{u=1}^N
\big[(K(o,u)(e_u-\pi_t(x_u)))_i - \langle \pi_t(x_o),K(o,u)(e_u-\pi_t(x_u))\rangle\big]\right\}.
\end{equation}

The following box is the main result. The state of our system, $Z$, now includes the model's probability distribution for the test point ($a$) and for *every single training point* ($b^{(v)}$). The theorem shows that this high-dimensional discrete process also converges to a coupled system of replicator ODEs. Each data point's prediction evolves according to its own replicator equation, but the fitness terms ($\phi$) are coupled, meaning the evolution of one depends on the current state of all the others, as one may expect.

::: theorem Continuous-Time Learning Dynamics of Full-Batch Training
Fix integers $V\ge 2$ and $N\ge 1$. For every index pair $(i,j)$ in $\{0,1,2,\dots,N\}$ fix a matrix $K(i,j)$ with $K(i,j)=K(j,i)^\top$. Define the uniform bound $\|K\|_{\max}:=\max_{o,u}\|K(o,u)\|_2$.

Define $a\in \Delta^{V-1}$ and a set $\{b^{(v)}\in \Delta^{V-1}\}_{v=1}^N$. Set the aggregate
\[
B:=(b^{(1)},\dots,b^{(N)})\in (\Delta^{V-1})^N,\qquad
Z:=(a,b^{(1)},\dots,b^{(N)})\in \underbrace{\Delta^{V-1}\times (\Delta^{V-1})^N}_{=:S}.
\]

*ODE system.* Let
\[
\phi^{(v)}(B):=\frac{1}{N}\sum_{u=1}^N K(v,u)\big[e_u-b^{(u)}\big],\qquad
\phi^{(a)}(B):=\frac{1}{N}\sum_{u=1}^N K(o,u)\big[e_u-b^{(u)}\big].
\]
The coupled replicator flow on $S$ is
\begin{equation*}
\begin{aligned}
\dot a_i &=a_i\Big(\phi^{(a)}_i(B)-\langle a,\phi^{(a)}(B)\rangle\Big),\\
\dot b^{(v)}_i &=b^{(v)}_i\Big(\phi^{(v)}_i(B)-\langle b^{(v)},\phi^{(v)}(B)\rangle\Big),\; v=1,\dots,N.
\end{aligned}\quad\text{(POP-ODE)}
\end{equation*}
With initial condition $Z(0)=Z_0\in S$ the solution is well-defined on $[0,T]$ for any $T>0$.

*Discrete scheme.* Given $\eta>0$ and $Z_0=Z(0)$, define recursively
\[
\tilde a_{n+1,i}=a_{n,i}\exp\Big\{\frac{\eta}{N}\sum_{u=1}^N \big[(K(o,u)(e_u-b^{(u)}_n))_i - a_n^\top K(o,u)(e_u-b^{(u)}_n)\big]\Big\},
\]
\[
\tilde b^{(v)}_{n+1,i}=b^{(v)}_{n,i}\exp\Big\{\frac{\eta}{N}\sum_{u=1}^N \big[(K(v,u)(e_u-b^{(u)}_n))_i - (b^{(v)}_n)^\top K(v,u)(e_u-b^{(u)}_n)\big]\Big\},
\]
\[
a_{n+1,i}=\frac{\tilde a_{n+1,i}}{\sum_j \tilde a_{n+1,j}},\qquad
b^{(v)}_{n+1,i}=\frac{\tilde b^{(v)}_{n+1,i}}{\sum_j \tilde b^{(v)}_{n+1,j}}.\quad\text{(POP-MU)}
\]

<p class="theorem-subtle">Then there exists a constant $C=C(T,Z_0,\|K\|_{\max})>0$ such that, for every $n$ with $t_n:=n\eta\le T$, we have $\|Z(t_n)-Z_n\|\le C|\eta|$ (Euclidean norm on $\R^{(N+1)\times V}$).</p>
:::
:::

## Learning Dynamics as a Polymatrix Replicator Game

The coupled system of ODEs we derived is a well-known structure in game theory called a **polymatrix replicator system**. Let's build the analogy piece by piece to see how our LLM learning dynamics fits perfectly into this framework.

In a standard replicator system, there's one mixed population. But in a polymatrix game, the population is divided into several distinct groups. The key feature is that the payoff for a strategy within one group depends on the strategy distributions across *all* groups. Meanwhile, the evolution of individuals within one group can be described by the same dynamics.

Here is the direct mapping from our learning problem to a polymatrix game:

- **Groups (or Populations) $\leftrightarrow$ Data Points:** Each data point in our system, including the test point $x_o$ and each training point $x_u$, corresponds to a separate group in the polymatrix game. We have $N+1$ groups in total.
- **Strategies $\leftrightarrow$ Labels:** For each group (data point), the available "strategies" are the $V$ possible labels the model can predict.
- **Population State $\leftrightarrow$ Predicted Probabilities:** The vector of predicted probabilities for a data point, $\pi_t(x_u)$, represents the "strategy mix" or population state for group $u$. For example, if $\pi_t(x_u) = [0.1, 0.7, 0.2]$, it's as if 10% of group $u$ is playing strategy 1 (label 1), 70% is playing strategy 2, and 20% is playing strategy 3.
- **Payoff Matrix $\leftrightarrow$ NTK Blocks:** The payoff for playing a strategy is determined by the NTK. The payoff for group $\alpha$ (e.g., data point $x_\alpha$) playing strategy $i$ (label $i$) depends on the strategies being played by all other groups $\beta$ through the kernel matrices $K(\alpha, \beta)$.

With this mapping, the ODE system from our theorem can be formally written as an instance of a polymatrix replicator game. For each data point $\alpha\in\{0,1,\dots,N\}$, we can define the evolution of its probability vector $x^{(\alpha)}$ as:
\[
\dot x^{(\alpha)}_i=\frac{1}{N}x^{(\alpha)}_i\Bigg[\underbrace{\sum_{\beta=1}^{N}\big(K(\alpha,\beta)[e_\beta-x^{(\beta)}]\big)_i}_{\text{Fitness of strategy } i \text{ for group } \alpha}
- \underbrace{\big\langle x^{(\alpha)},\;\sum_{\beta=1}^{N}K(\alpha,\beta)[e_\beta-x^{(\beta)}]\big\rangle}_{\text{Average fitness for group } \alpha}\Bigg]
\]

This equation is precisely the replicator dynamic: the growth rate of a strategy (label probability) is proportional to its current share and its fitness advantage over the average.

::: item An Intuition for Fitness
What does "fitness" mean in the context of LLM learning? The fitness term for a given label $i$ at a data point $\alpha$ is a sum of interactions with all training data points $\beta \in \{1, ..., N\}$. Each interaction, governed by the kernel $K(\alpha, \beta)$, creates a "push-pull" dynamic.

Let's break down the fitness term for label $i$ at data point $\alpha$:
\[
\text{Fitness}_i^{(\alpha)} = \frac{1}{N}\sum_{\beta=1}^N \Big( \underbrace{\langle \nabla f_i(\alpha), \nabla f_{y_\beta}(\beta) \rangle}_{\text{Pull towards true label}} - \underbrace{\mathbb{E}_{j \sim \pi(\cdot|\beta)} \left[ \langle \nabla f_i(\alpha), \nabla f_j(\beta) \rangle \right]}_{\text{Push from current prediction}} \Big)
\]

Here, we've expanded the kernel $K_{ij}(\alpha,\beta)$ as the inner product of Jacobians $\langle\nabla f_i(\alpha),\nabla f_j(\beta)\rangle$.

- The **"Pull" Term** measures the alignment (or synergy) between the gradient for label $i$ at point $\alpha$ and the gradient for the *true label* $y_\beta$ at point $\beta$. High alignment means that increasing the probability of label $i$ for $\alpha$ is "helpful" for correctly classifying $\beta$. This term pulls the dynamics towards the ground truth data.
- The **"Push" Term** measures the average alignment between the gradient for label $i$ at $\alpha$ and the gradients for all labels at $\beta$, weighted by the model's *current prediction* $\pi(\cdot|\beta)$. This term represents the influence of the model's own beliefs. If the model is currently confident in a wrong answer for $\beta$, this term might push the dynamics in the wrong direction.

Learning is thus a dynamic competition. The probability of a label grows if, on average, it aligns better with the true labels across the dataset than it does with the model's current (and potentially flawed) predictions.
:::

---
