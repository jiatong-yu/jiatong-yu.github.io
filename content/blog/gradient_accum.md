---
title: The Implicit Bias of Gradient Accumulation in RLHF
tldr: This blog explains where likelihood displacement comes from, by viewing training as a continuous-time flow. Then, by taking Lie–Trotter splitting of the same gradient flow, we arrived at an alternative RL training algorithm that corrects the flow's bias and mitigates likelihood displacement effects.
author: Jiatong Yu
date: September 2026
description: The implicit bias of gradient accumulation in RLHF
---

\tableofcontents

## Overview
Reward and preference-based algorithms used for RL training share a similar gradient structure backbone:

\[
\nabla_\theta \mathcal{L}(\theta) = - \mathbb{E}_{x,y} \big[
    \alpha_y \cdot \nabla_\theta \log \pi_\theta (y \mid x)
     \big], \quad \alpha_y \in \mathbb{R}
\tag{1}
\]
where $\alpha_y$ is a scalar reward on the completion $y$. The scalar's sign splits every batch into two groups:
- $\alpha_y > 0$: a descent step on the *preferred output*. In this blog, we label trajectories with positive reward as $y^+$.
- $\alpha_y < 0$: an ascent step on the *dispreferred output*, with trajectories labeled as $y^-$.

The training objective is to push up the likelihood of $y^+$ and push down the likelihood of $y^-$, but this is not always the case in practice. [*Likelihood displacement*](https://arxiv.org/html/2410.08847v4) refers to when training pushes down the likelihood of *both* $y^+$ and $y^-$. [Later work](https://arxiv.org/html/2410.13828v2) attributes this to *gradient entanglement* — the inner product between the gradients of $y^+$ and $y^-$ can be very large, leading to synchronized likelihood changes.

This blog explains where gradient entanglement comes from, by viewing training as a continuous-time flow. Then, by splitting the same gradient flow, we arrived at an alternative margin-based training algorithm that corrects the flow's bias and mitigates likelihood displacement effects.

We test this with 7B models on DPO. We observe that the proposed algorithm regularizes against gradient entanglement, and largely prevents likelihood displacement.

## Setup

Let $\pi_\theta(y\mid x)$ be the probability of completion $y$ given prompt $x$, and write its negative log-likelihood as

$$
\ell_\theta(x,y)=-\log\pi_\theta(y\mid x).
$$

On a batch of $N$ examples, let $\alpha_i$ be a signed weight. Positive weights encourage higher likelihood; negative weights encourage lower likelihood. Define

$$
L^+(\theta)
=\frac{1}{N}\sum_{i:\alpha_i>0}\alpha_i\ell_\theta(x_i,y_i),
\qquad
L^-(\theta)
=\frac{1}{N}\sum_{i:\alpha_i<0}\alpha_i\ell_\theta(x_i,y_i),
$$

$$
\mathcal L=L^++L^-.
$$

The minus-group contribution $L^-$ is signed: $\textcolor{red}{-}L^-$ is the negative log-likelihood of the $y^-$ batch. Define the gradeint symbols:

$$
g^+=\nabla L^+,
\qquad
g^-=-\nabla L^-,
$$

where $-g^+$ and $-g^-$ point to the direction of increased likelihood for $y^+$ and $y^-$, respectively. (Make sure you understand the signs and symbols here!)

The gradient used by the training objective is their *difference*:

$$
\nabla\mathcal L=g^+-g^-.
$$

We begin with SGD's ordinary gradient step, conditioned on a fixed batch. The derivations treat the examples, weights, and group assignments as fixed, with smooth component losses. This is a deterministic surrogate for the update, not a complete analysis of minibatch noise. For a batch, likelihood statements refer to weighted group NLLs and need not hold for every response individually. The additional assumptions needed for Adam's trajectory-level reduction are stated when we reach it.

## Penalizing Gradient Entanglement in SGD

The gradient flow of the signed objective is

$$
\dot\theta=-\nabla\mathcal L=-g^++g^-.
$$

The margin-based training is a discretization of the flow:

$$
\theta_{\mathrm{agg}}
=\theta-h\bigl(g^+(\theta)-g^-(\theta)\bigr).
$$

It is well-known that, with finite step size $h$, the training dynamics follow a modified loss:

$$
\widetilde{\mathcal L}_{\mathrm{agg}}
=\mathcal L+\frac{h}{4}\|g^+-g^-\|^2.
$$

This is the implicit gradient regularization result of [Barrett and Dherin][barrett], applied to the margin-based objective

Expanding the squared norm:

$$
\widetilde{\mathcal L}_{\mathrm{agg}}
=
\mathcal L
+\underbrace{\frac{h}{4}
\bigl(\|g^+\|^2+\|g^-\|^2\bigr)}_{\text{individual gradient penalties}}
-\underbrace{\frac{h}{2}\langle g^+,g^-\rangle}_{\text{alignment reward}}.
$$

Thus the modified loss identifies a finite-step bias that we can try to remove.

### Splitting Scheme Removes the Bias for Gradient Entanglement

Consider, instead of combining both gradients at their original values, apply the positive contribution first and recompute the negative contribution afterward:

$$
\theta_1=\theta-hg^+(\theta),
\qquad
\theta_{+\to-}=\theta_1+hg^-(\theta_1).
$$

The reverse order is equally natural: first ascend the negative-group NLL, then descend the positive-group NLL at the shifted parameters. Crucially, the second gradient is *recomputed*. 

A fixed ordering introduces an extra order-dependent term. To study the effect of splitting in isolation, imagine an ideal algorithm that chooses the order with a fresh fair coin before every pair. Averaging over that coin cancels the leading order-dependent term. The resulting leading conditional-mean modified loss is

$$
\widetilde{\mathcal L}_{\mathrm{split}}
=\mathcal L+\frac{h}{4}
\bigl(\|g^+\|^2+\|g^-\|^2\bigr).
$$

Comparing it with aggregation gives the central identity:

$$
\boxed{
\widetilde{\mathcal L}_{\mathrm{split}}
-\widetilde{\mathcal L}_{\mathrm{agg}}
=\frac{h}{2}\langle g^+,g^-\rangle.
}
$$

Splitting removes aggregation's alignment reward. Relative to the baseline margin-based objective, that is an *inner-product penalty*.

### Experiments

We test the proposed alignment-penalizing algorithm with Qwen2.5-7B-Instruct with DPO using plain gradient descent. For convenience, we call the baseline margin-based algorithm the *aggregate arm*, while the proposed alignment-penalizing algorithm the *sequential arm*.

**Setup**. We take DPO $\beta = 0.1$ with $64$ pairs per batch. Four H100, fp32 master weights. The sequential arm takes two half-learning-rate steps compared to the aggregate arm. We take the preference pairs from GSM8K, where $y^+$ comes from reference solutions and $y^-$ comes from contaminated solutions swapped with wrong answers.

**Sequential update mitigates likelihood displacement**. 

As shown in Figure 1, for semantically similar pairs, we see that standard DPO exhibit likelihood displacement, where the log likelihood of preferred outputs decreased monotonically during training. On the other hand, the proposed sequential arm largely mitigates the displacement. On held-out preference pairs, baseline likelihood displacement exists but the sequential arm's mitigation strength is weaker. As expected, the DPO margins between the two algorithms are almost identical along training.

![Main result](assets/gradient_accum/fig_main.png "Figure 1: Baseline DPO suffers from likelihood displacement, proposed sequential arm mitigates it.")

**Shared text between preference pairs drives the result**. 

Define $c = \tfrac12\,(g^{+} + g^{-})$, $d= \tfrac12\,(g^{+} - g^{-})$.
Call $c$ the *common mode* and $d$ the *differential mode*. The names come from where the two gradients get their overlap. The gradient of an output's log-likelihood is a sum of per-token terms, and a token that two outputs share, in the same context, contributes the same term to both. When preferred and dispreferred outputs overlap in their tokens, $g^{+}$ and $g^{-}$ therefore share a large component; $c$ is that shared component and $d$ is what distinguishes them. 

Figure 2 decomposes the inner product into different components. As expected, our proposed algorithm regularizes gradient entanglement by an order of magnitude. Interestingly, the inner product gap is driven by the common mode (e.g. gradients over the shared text between $y^+$ and $y^-$) gradient norm and the angle between the gradient directions. 

![Gradient modes](assets/gradient_accum/fig_modes.png "Figure 2: Proposed sequential arm regularizes inner product between $g^+$ and $g^-$.")

**Ablations**. 

Our proposed sequential arm splits one preference batch into the preferred output batch $y^+$ and dispreferred output batch $y^-$, and perform two gradient update steps on each.
Our theory relies on an idealized algorithm: for each iteration, we randomly choose to start with $y^+$ batch or $y^-$ batch. This is to remove a Lie bracket term that makes analysis annoying. In ablations, we primarily study whether we can relax this random coin flips per iteration and train on a deterministic schedule.

As shown in Figure 1, in small learning rates, the ordering of whether to start with $y^-$ batch or $y^+$ batch does not matter. Whether to do the random coin flip, or always start with one batch then another are the same.

![Controls](assets/gradient_accum/fig_ablations.png "Figure 3: Ablation studies over the ordering of sequential update stages.")

At higher learning rates, the ordering of stages starts to matter. As shown in Figure 4, sequential update that always start with $y^+$ is much better than its counterpart that always start with $y^-$: it's able to mitigate likelihood displacement while pushing up the preference margin.

Why does starting with $y^-$ hurt performance so much? I hypothize this is a mechanism about gradient norm that the theory can't see. In Figure 4 (c), we observe that $\|g^-\|$ is several orders of magnitude larger than $\|g^-\|$. Taking a gradient step along this direction drastically changes the likelihood of shared text between $y^+$ and $y^-$, the damage of which is not recoverable in training.

![Stability](assets/gradient_accum/fig_stability.png "Figure 4: Sequential arm is less stable at higher learning rate.")

## Penalizing Gradient Entanglement in Adam

SGD is rarely used in practice, so we proceed to investigate how to mitigate gradient entanglement issues with AdamW as the optimizer. As a primer, let's recall the Adam update. Define 
$$
q_n=g^+(\theta_n)-g^-(\theta_n).
$$

The baseline algorithm is

$$
\begin{aligned}
m_{n+1}&=\beta_1m_n+(1-\beta_1)q_n,\\
v_{n+1}&=\beta_2v_n+(1-\beta_2)q_n^{\odot2},\\
\widehat m_{n+1}&=\frac{m_{n+1}}{1-\beta_1^{n+1}},
\qquad
\widehat v_{n+1}=\frac{v_{n+1}}{1-\beta_2^{n+1}},\\
\theta_{n+1}&=\theta_n-h
\frac{\widehat m_{n+1}}{\sqrt{\widehat v_{n+1}}+\epsilon}.
\end{aligned}
$$

We use the framework proposed by [Shigida et al.](https://papers.nips.cc/paper_files/paper/2025/hash/e4cc8ab4a64e99f962f36d07a7723d94-Abstract-Conference.html) The idea is to expand past gradients around the current parameters, replacing the dependence on history by a current-parameter correction. Backward error analysis can then be applied to that approximate update. This is an analytical operation; the implemented optimizer still maintains $m$ and $v$.

**Assumptions**. Suppose fixed $\beta_1,\beta_2<1$, and a small constant step size, after initialization effects have become negligible. Gradients must vary slowly over the effective memory window. For this statement, take $\epsilon=0$ and restrict attention to a region where every coordinate of $q$ is bounded away from zero.

**Gradient Flow**. In that regime the leading flow is

$$
\dot\theta=-P(\theta)\nabla\mathcal L(\theta),
\qquad
P(\theta)=\operatorname{diag}
\left(\frac{1}{|q_1(\theta)|},\ldots,
\frac{1}{|q_d(\theta)|}\right),
$$

where $d$ is the number of trainable parameters. Accounting for memory gives the baseline modified loss:

$$
\boxed{
\widetilde{\mathcal L}_{\mathrm{agg}}^{\mathrm{Adam}}
=
\mathcal L
+h\left(
\frac{\beta_1}{1-\beta_1}
-\frac{\beta_2}{1-\beta_2}
\right)
\|g^+-g^-\|_1.
}
$$

Its modified flow is $-P\nabla\widetilde{\mathcal L}_{\mathrm{agg}}^{\mathrm{Adam}}+O(h^2)$. The norm $\|q\|_1=\sum_j|q_j|$ is the sum of absolute coordinate values, not a squared Euclidean norm (see Appendix C for rederivations).

When $\beta_2>\beta_1$, its coefficient is negative (despite rare in practice). Therefore we do *not* hold any parallel claim about baseline margin-based training encourages entanglement, in contrast to the SGD results. Our aim is instead to produce an implicit bias that punishes the inner-product term relative to Adam's own baseline.

### Virtual Splitting Scheme

Now we introduce our proposed sequential arm algorithm for Adam. In short, the first moment receives gradients evaluated after small steps of the opposite group. The second moment continues to receive the original aggregate gradient.

Let $\kappa > 0$ be a hyperparameter of the proposed algorithm; keep it fixed as $h$ decreases.
At the start of an outer step, compute $g^+(\theta_n)$, $g^-(\theta_n)$, and their difference $q_n$ as defined above. Then define two *virtual evaluation points*:

$$
\boxed{
\theta_{+,n}
=\theta_n+\frac{\kappa h}{2}g^-(\theta_n),
\qquad
\theta_{-,n}
=\theta_n-\frac{\kappa h}{2}g^+(\theta_n).
}
$$

The positive loss is evaluated after a virtual negative-group update; the negative loss is evaluated after a virtual positive-group update. Accumulate

$$
q_n^{\mathrm{stage}}
=g^+(\theta_{+,n})-g^-(\theta_{-,n}).
$$

Note that we do *not* differentiate through the construction of the virtual points, and neither point is committed as the new model. The single outer update is

$$
\begin{aligned}
m_{n+1}
&=\beta_1m_n+(1-\beta_1)
\underbrace{q_n^{\mathrm{stage}}}_{\text{crossed evaluations}},\\
v_{n+1}
&=\beta_2v_n+(1-\beta_2)
\underbrace{q_n^{\odot2}}_{\text{original aggregate gradient}},\\
\theta_{n+1}
&=\theta_n-h
\frac{\widehat m_{n+1}}{\sqrt{\widehat v_{n+1}}+\epsilon}.
\end{aligned}
$$

The bias corrections are the same as in baseline Adam. The optimizer clock advances once. At $\kappa=0$, the update reduces to aggregation.
A first-order expansion of the two evaluations gives

$$
q_n^{\mathrm{stage}}
=
\nabla\left[
\mathcal L+
\frac{\kappa h}{2}\langle g^+,g^-\rangle
\right](\theta_n)
+O(h^2).
$$

The two cross-effects add to the gradient of the inner product. Appendix B gives the expansion and explains exactly what is differentiated.

Accounting for Adam's memory, under the same deterministic fixed-$\beta$ assumptions as the baseline, yields

$$
\boxed{
\widetilde{\mathcal L}_{\mathrm{stage}}^{\mathrm{Adam}}
=
\widetilde{\mathcal L}_{\mathrm{agg}}^{\mathrm{Adam}}
+
\underbrace{\frac{\kappa h}{2}
\langle g^+,g^-\rangle}_{\text{implicit alignment penalty}}.
}
$$



## Appendix

The derivations behind every formula above — the modified losses, the memory elimination for Adam, the finite-$\epsilon$ and moment-retaining variants, the compute accounting, and the synthetic checks — are collected in [the appendix](gradient_accum_appendix.html).

[barrett]: https://arxiv.org/abs/2009.11162