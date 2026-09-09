---
title: Appendix — The Implicit Bias of Gradient Accumulation in RLHF
tldr: Derivations, scope conditions, and synthetic checks supporting the main post.
author: Jiatong Yu
date: September 2026
description: Appendix to the implicit bias of gradient accumulation in RLHF
---

Appendix to [The Implicit Bias of Gradient Accumulation in RLHF](gradient_accum.html).

The appendices retain the derivations needed to check signs, coefficients, and scope. The SGD formulas specialize existing backward-error results. The crossed-stage Adam construction and its relative alignment term are the proposed extension developed in this note.

## Appendix A. Gradient flow and the SGD modified losses

For compactness in the proofs only, write

$$
a=g^+,
\qquad b=g^-,
\qquad q=a-b,
\qquad C=a^\top b,
$$

$$
H_+=Dg^+=\nabla^2L^+,
\qquad H_-=Dg^-=-\nabla^2L^-,
\qquad H=H_+-H_-.
$$

Here a Hessian, such as $H_+$, describes how a gradient changes when the parameters change. The component Hessians are symmetric, but need not be positive semidefinite. In particular,

$$
\nabla C=H_+b+H_-a.
$$

All right-hand-side quantities below are evaluated at the common starting parameters unless otherwise stated. Assume sufficiently smooth losses with derivatives bounded on the neighborhood under consideration, so that the displayed Taylor remainders are uniform.

### A.1 The individual likelihood rates

Since $\dot\theta=-a+b$, the chain rule gives

$$
\frac{dL^+}{dt}=-\|a\|^2+C,
\qquad
\frac{d(-L^-)}{dt}=\|b\|^2-C.
$$

Both group NLLs rise when $\|a\|^2<C<\|b\|^2$. To decrease positive NLL while increasing negative NLL, the required inequalities are

$$
C\leq\min\{\|a\|^2,\|b\|^2\}.
$$

These formulas diagnose the original flow; they do not yet depend on a discretization.

### A.2 How to obtain a modified equation

Suppose a numerical map has expansion

$$
\theta^+=\theta+hf(\theta)+h^2r(\theta)+O(h^3).
$$

The time-$h$ flow of $\dot\theta=f+h f_1$ expands as

$$
\theta+hf+h^2\left(f_1+\frac12Df\,f\right)+O(h^3).
$$

Matching the two maps therefore gives

$$
\dot\theta=f+h\left(r-\frac12Df\,f\right)+O(h^2).
$$

For aggregation, $f=-q$, $r=0$, and $Df\,f=Hq$. Since $\nabla\|q\|^2=2Hq$,

$$
\dot\theta
=-q-\frac h2Hq+O(h^2)
=-\nabla\left(\mathcal L+\frac h4\|q\|^2\right)+O(h^2).
$$

This is the application of [Barrett and Dherin][barrett] used in the main text.

### A.3 Splitting and the order-dependent term

The two ordered maps satisfy

$$
\begin{aligned}
\theta_{+\to-}
&=\theta-hq-h^2H_-a+O(h^3),\\
\theta_{-\to+}
&=\theta-hq-h^2H_+b+O(h^3).
\end{aligned}
$$

For the preferred-first ordering, the modified field is

$$
\dot\theta
=-q-\frac h2Hq-hH_-a+O(h^2).
$$

Rearranging it gives

$$
\dot\theta
=-\nabla\left[
\mathcal L+\frac h4(\|a\|^2+\|b\|^2)
\right]
+\frac h2\underbrace{(H_+b-H_-a)}_{\text{order-dependent term}}
+O(h^2).
$$

Using the convention $[f_+,f_-]=(Df_-)f_+-(Df_+)f_-$ for the actual update fields $f_+=-a$ and $f_-=b$, the displayed order term is their Lie bracket. Its sign reverses in the opposite ordering. This matches the sequential-update structure analyzed by [Dherin][dherin]. A fixed-order method therefore generally needs this extra field as well as the scalar modified loss.

A fresh fair choice cancels the bracket in the conditional mean. Directly at the map level,

$$
\mathbb E_{\mathrm{order}}[\theta_{\mathrm{split}}\mid\theta]
=\theta_{\mathrm{agg}}-\frac{h^2}{2}\nabla C+O(h^3).
$$

The scalar difference can also be written in the form

$$
\begin{aligned}
\widetilde{\mathcal L}_{\mathrm{split}}
-\widetilde{\mathcal L}_{\mathrm{agg}}
&=\frac h8\left(\|a+b\|^2-\|a-b\|^2\right)\\
&=\frac h2C.
\end{aligned}
$$

Finally, Taylor expansion of $C$ after each map gives

$$
\boxed{
\mathbb E_{\mathrm{order}}[C(\theta_{\mathrm{split}})\mid\theta]
-C(\theta_{\mathrm{agg}})
=-\frac{h^2}{2}\|\nabla C(\theta)\|^2+O(h^3).
}
$$

The leading coefficient is nonpositive and strictly negative when $\nabla C\neq0$. The comparison is from a common starting point. It does not order entire trajectories, and the random maps retain variance beyond their mean drift.

## Appendix B. What the crossed loss evaluations generate

Set $\rho=\kappa h$, with fixed $\kappa$. The deterministic staged gradient is

$$
q^{\mathrm{stage}}
=a\left(\theta+\frac\rho2b\right)
-b\left(\theta-\frac\rho2a\right).
$$

Expanding the component gradients,

$$
\begin{aligned}
a\left(\theta+\frac\rho2b\right)
&=a+\frac\rho2H_+b+O(\rho^2),\\
b\left(\theta-\frac\rho2a\right)
&=b-\frac\rho2H_-a+O(\rho^2).
\end{aligned}
$$

Therefore

$$
\boxed{
q^{\mathrm{stage}}
=q+\frac\rho2\nabla C+O(\rho^2).
}
$$

There are no self-gradient terms $H_+a$ or $H_-b$ in this relative correction. For quadratic component losses, the identity is exact because their gradients are affine. At finite $\rho$, the staged field need not itself be exactly conservative; the scalar claim is to the displayed order.

### B.1 The actual step-dependent loss

At iteration $n$, freeze $a_n=a(\theta_n)$ and $b_n=b(\theta_n)$, and define

$$
J_n(z)
=L^+\left(z+\frac\rho2b_n\right)
+L^-\left(z-\frac\rho2a_n\right).
$$

Differentiating with respect to $z$, while treating $a_n,b_n$ as constants, gives

$$
\nabla_zJ_n(z)\big|_{z=\theta_n}=q_n^{\mathrm{stage}}.
$$

This is the computation meant by evaluating the original losses at crossed stages. It requires only first-order differentiation. Differentiating through $a_n,b_n$ would define a different algorithm.

There is a subtle distinction between the value of this step-dependent loss and the modified loss of the algorithm. Substituting $z=\theta_n$ and expanding values gives $J_n(\theta_n)=\mathcal L(\theta_n)+\rho C(\theta_n)+O(\rho^2)$. Its derivative with the displacements held fixed contains $\rho\nabla C/2$, not $\rho\nabla C$. The modified loss must be inferred from that actual derivative and the update, not by differentiating the value expansion as if the frozen displacements varied with $z$.

### B.2 Why using the same staged input twice can cancel the effect

Write $q^{\mathrm{stage}}=q+hr+O(h^2)$, with $r=\kappa\nabla C/2$. In the memoryless case, using this gradient in both Adam moments gives

$$
\frac{q_j+hr_j}{|q_j+hr_j|+\epsilon}
=
\frac{q_j}{|q_j|+\epsilon}
+h\frac{\epsilon}{(|q_j|+\epsilon)^2}r_j
+O(h^2)
$$

on a region with unchanged signs. At $\epsilon=0$, the first-order response vanishes. With the original-point denominator, the response is instead $hr_j/(|q_j|+\epsilon)$. Appendix C shows the same cancellation mechanism after memory is included.

### B.3 Why two ordinary Adam steps are not the same construction

Even with parameters held fixed, feeding the two signed inputs $a$ and $-b$ into separate second-moment updates produces

$$
\begin{aligned}
v_{+\to-}
&=\beta_2^2v+(1-\beta_2)(\beta_2a^{\odot2}+b^{\odot2}),\\
v_{-\to+}
&=\beta_2^2v+(1-\beta_2)(a^{\odot2}+\beta_2b^{\odot2}).
\end{aligned}
$$

Their fair average contains separate squares, not the aggregate square

$$
(a-b)^{\odot2}
=a^{\odot2}+b^{\odot2}-2a\odot b.
$$

Randomizing the order does not restore the missing coordinatewise cross term, and also does not undo the extra advance of the moment clock. Cross-staged Adam avoids that leading change by taking one outer step with the original aggregate second-moment input.

## Appendix C. Memory elimination and the scalar Adam result

This appendix uses the approach of [Cattaneo and Shigida][memory] and recovers the baseline correction in [Cattaneo, Klusowski, and Shigida][adam-bias] before adding the staged contribution.

Assume a fixed smooth deterministic surrogate, constant $h$, fixed $\beta_1,\beta_2<1$, and bounded derivatives on the region visited. Work with $\epsilon=0$ and $|q_j|\geq c>0$ for every coordinate, with $c$ independent of $h$. Moment initialization and bias-correction remainders must either be retained or be smaller than the error order being claimed. Equivalently, the moments can be initialized consistently with the expansions. Merely calling an iteration "late" is not sufficient if a remaining transient is appreciable.

Use the appendix notation from Appendix A, and define

$$
P=\operatorname{diag}(1/|q_j|),
\qquad
\tau_i=\frac{\beta_i}{1-\beta_i},
\qquad
\chi=\tau_1-\tau_2,
\qquad
r=\frac\kappa2\nabla C.
$$

The staged input is $q+hr+O(h^2)$. To leading order, the backward parameter offsets satisfy

$$
\theta_{n-k}=\theta_n+khPq+O(k^2h^2).
$$

The exponential memory weights make the corresponding sums of Taylor remainders finite for fixed $\beta_i<1$. Constants deteriorate when a momentum parameter approaches one; the regime requires small changes across the memory window, not just a small nominal learning rate.

### C.1 Expand the two moments

Ignoring only the initialization terms specified above, the first moment can be written as

$$
m_{n+1}
=(1-\beta_1)\sum_{k\geq0}\beta_1^k
q^{\mathrm{stage}}(\theta_{n-k}).
$$

Since

$$
(1-\beta)\sum_{k\geq0}\beta^k=1,
\qquad
(1-\beta)\sum_{k\geq0}k\beta^k=\frac\beta{1-\beta},
$$

its expansion is

$$
\boxed{
m_{n+1}=q+h\tau_1HPq+hr+O(h^2).
}
$$

The original-point second-moment input gives

$$
\boxed{
v_{n+1}=q^{\odot2}
+2h\tau_2q\odot(HPq)+O(h^2).
}
$$

There is no $2h q\odot r$ term in the second line. If the staged gradient were used there as well, that term would appear and would cancel the added numerator contribution at this order when $\epsilon=0$.

The coefficient of $hr$ in the first moment is one, not $1-\beta_1$: the expansion includes the staged inputs throughout the memory history. The immediate response from an arbitrary fixed state is a different comparison, given in Appendix E.

### C.2 Form the parameter update

Expanding the coordinatewise ratio yields

$$
\frac{m_{n+1}}{\sqrt{v_{n+1}}}
=Pq+h\chi PHPq+hPr+O(h^2).
$$

Thus the memoryless approximation to the implemented map is

$$
\theta_{n+1}
=\theta_n-hPq-h^2\chi PHPq-h^2Pr+O(h^3).
$$

On a region with fixed gradient signs, $Pq=\operatorname{sign}(q)$ is locally constant. Consequently the leading field has $Df\,f=0$, and there is no additional order-$h$ term from the map-to-flow conversion in Appendix A. The modified equation is

$$
\dot\theta=-Pq-h\chi PHPq-hPr+O(h^2).
$$

Because $H$ is symmetric,

$$
\nabla\|q\|_1=H\operatorname{sign}(q)=HPq.
$$

Substituting $r=\kappa\nabla C/2$ gives

$$
\boxed{
\dot\theta
=-P\nabla\left[
\mathcal L+h\chi\|q\|_1+\frac{\kappa h}{2}C
\right]+O(h^2).
}
$$

Setting $\kappa=0$ recovers aggregate Adam. The equality characterizes the leading modified field in the stated geometry; it is not an exact all-orders identity or a general convergence theorem.

## Appendix D. Finite stabilizer and moment-retaining descriptions

### D.1 The finite-$\epsilon$ modified field

For the outside-square-root convention used in the algorithm, let

$$
P_\epsilon=\operatorname{diag}\left(\frac1{|q_j|+\epsilon}\right),
\qquad
s_j=\frac{|q_j|}{|q_j|+\epsilon}.
$$

Under the deterministic fixed-$\beta$ assumptions, still locally away from vanishing gradient coordinates for the square-root expansion, memory elimination gives

$$
\theta_{n+1}
=\theta_n-hP_\epsilon q
-h^2P_\epsilon\operatorname{diag}(\tau_1-\tau_2s_j)HP_\epsilon q
-h^2P_\epsilon r+O(h^3).
$$

Here the map-to-flow conversion is no longer zero. For $f=-P_\epsilon q$,

$$
Df\,f=P_\epsilon\operatorname{diag}(1-s_j)HP_\epsilon q.
$$

Define

$$
K=\frac12\operatorname{diag}\left(
\frac{1+\beta_1}{1-\beta_1}
-\frac{1+\beta_2}{1-\beta_2}s_j
\right).
$$

The resulting modified field is

$$
\dot\theta
=-P_\epsilon q-hP_\epsilon KHP_\epsilon q
-\frac{\kappa h}{2}P_\epsilon\nabla C+O(h^2).
$$

Therefore the relative contribution remains

$$
\boxed{
\widetilde f_{\mathrm{stage}}
-\widetilde f_{\mathrm{agg}}
=-P_\epsilon\nabla\left(\frac{\kappa h}{2}C\right)+O(h^2).
}
$$

Why not always write the whole field as a modified loss? Define

$$
R_\epsilon(\theta)
=\sum_j\left[
|q_j|-\epsilon\log\left(1+\frac{|q_j|}{\epsilon}\right)
\right].
$$

Then $\nabla R_\epsilon=HP_\epsilon q$, but $K\nabla R_\epsilon$ need not be a gradient: multiplication by coordinate-dependent coefficients does not generally preserve equality of mixed partial derivatives. The added alignment term is identifiable even when the rest of the field has no scalar representation in the prescribed metric.

The cited Adam analyses also use a stabilizer inside the square root in parts of their presentation. The finite-$\epsilon$ expressions here are derived for $\sqrt v+\epsilon$; those conventions should not be interchanged silently. Both approach the displayed zero-stabilizer scalar form only under the corresponding nonvanishing-gradient assumptions.

### D.2 When the moments remain dynamical variables

A different limit takes $\beta_i=1-\gamma_i h$ with fixed $\gamma_i>0$, so memory persists on the same time scale as the parameters. The deterministic post-transient leading system is

$$
\dot\theta=-P(v)m,
\qquad
\dot m=\gamma_1(q-m),
\qquad
\dot v=\gamma_2(q^{\odot2}-v),
$$

where $P(v)=\operatorname{diag}(1/(\sqrt{v_j}+\epsilon))$. This is the moment-retaining type of description developed by [Barakat and Bianchi][adam-ode]; bias correction requires its time dependence to be retained as well.

Our staged input adds $\kappa h\nabla C/2+O(h^2)$ to $q$. Consequently, relative to the baseline modified system, the order-$h$ change in the joint field is

$$
\begin{pmatrix}
\Delta\dot\theta\\
\Delta\dot m\\
\Delta\dot v
\end{pmatrix}
=
\begin{pmatrix}
0\\
\dfrac{\kappa h\gamma_1}{2}\nabla C\\
0
\end{pmatrix}
+O(h^2).
$$

The alignment force first enters the momentum equation; it does not immediately appear as the reduced parameter drift from Appendix C. The two limits must not be conflated by substituting $\beta_i=1-\gamma_i h$ into a fixed-$\beta$ expansion whose constants then diverge.

### D.3 Stochastic and implementation scope

For a stochastic aggregate gradient $G=q+\xi$ with conditional mean-zero noise,

$$
\mathbb E[G^{\odot2}\mid\theta]
=q^{\odot2}+\mathbb E[\xi^{\odot2}\mid\theta].
$$

Adam's denominator therefore responds to noise as well as the mean gradient. The deterministic $\|q\|_1$ surrogate is not automatically a population-level modified loss for minibatch Adam. Random-order expectations in this note average only over the order coin, holding the batch and starting state fixed.

Similarly, model-dependent weights or sampling distributions change the derivatives if they are recomputed. The within-step staged-gradient identity can be applied to a frozen surrogate, but its fixed-loss memory analysis does not automatically apply across a sequence of changing surrogates. Clipping, weight decay, stochastic layers, and learning-rate schedules also require the actual implemented map to be specified. None is silently included in the scalar formulas above.

## Appendix E. Random-order staging and a direct alignment comparison

The two full-displacement stage gradients expand as

$$
\begin{aligned}
q_{+\to-}^{\mathrm{seq}}
&=q+\rho H_-a+O(\rho^2),\\
q_{-\to+}^{\mathrm{seq}}
&=q+\rho H_+b+O(\rho^2).
\end{aligned}
$$

For a fresh fair coin,

$$
\mathbb E_{\mathrm{order}}[q^{\mathrm{seq}}\mid\theta]
=q+\frac\rho2\nabla C+O(\rho^2).
$$

This agrees with the symmetric half-displacement construction to the displayed order, not necessarily at finite $\rho$. The first-moment update is linear in this input, and the original-point denominator is identical for both branches when starting from the same optimizer state.

To make the local comparison without eliminating memory, fix arbitrary common $(\theta_n,m_n,v_n)$ and the same batch. Let

$$
P_n=\operatorname{diag}\left(
\frac1{\sqrt{\widehat v_{n+1,j}}+\epsilon}
\right),
\qquad
c_n=\frac{1-\beta_1}{1-\beta_1^{n+1}},
$$

where $v_{n+1}$ is the shared original-gradient update. For the deterministic crossed stages,

$$
\theta_{\mathrm{stage}}-\theta_{\mathrm{agg}}
=-hc_nP_n(q^{\mathrm{stage}}-q)
=-\frac{\kappa h^2}{2}c_nP_n\nabla C+O(h^3).
$$

Assuming the baseline displacement is $O(h)$ and $P_n$ is locally bounded, Taylor expansion gives

$$
\boxed{
C(\theta_{\mathrm{stage}})-C(\theta_{\mathrm{agg}})
=-\frac{\kappa h^2}{2}c_n
\nabla C^\top P_n\nabla C+O(h^3).
}
$$

The randomized construction satisfies the same formula with conditional expectation on the left. The leading coefficient is nonpositive because $c_n>0$ and $P_n$ is positive definite. No sign assumption on either component Hessian is needed.

The factor $c_n$ is the immediate sensitivity of the first moment to a new input. In the fixed-$\beta$ reduced modified loss, the coefficient instead includes the accumulation of staged inputs across history, as derived in Appendix C. These are different, consistent comparisons. Neither says that moment states or parameter trajectories remain equal after the two algorithms begin taking different updates.

## Appendix F. Compute, memory, and global-batch implementation

For the preferred-first randomized branch, the required component evaluations are

$$
a(\theta),\qquad b(\theta),\qquad b(\theta-\rho a(\theta)).
$$

The reversed branch reevaluates $a$ instead. Their expected work is

$$
\frac12(W_++2W_-)+\frac12(2W_++W_-)
=\frac32(W_++W_-).
$$

This assumes the component gradients can be collected without repeating a full joint-batch backward. Masking losses and running multiple backwards through the same concatenated graph does not by itself guarantee that assumption.

The moment and parameter-update arithmetic remains linear in the number of trainable parameters. Persistent state is still just $m,v$. With sequential graph release there is no requirement to retain all activation graphs at once. A carefully streamed randomized implementation can use an additional gradient-sized direction buffer beyond the ordinary gradient buffer, but reliable parameter restoration, mixed-precision overflow handling, and temporary optimizer operations can require more. An FP32 vector over $N$ trainable parameters occupies $4N$ bytes before sharding. These are working-memory estimates, not a measured peak-memory result.

For $K$ equally weighted microbatches, the intended batch-level alignment is

$$
\left(\frac1K\sum_i a_i\right)^\top
\left(\frac1K\sum_j b_j\right)
=\frac1{K^2}\sum_{i,j}a_i^\top b_j.
$$

Independent microbatch-local stages generally target the average of $a_i^\top b_i$, omitting cross-microbatch terms. The same issue applies across data-parallel workers. Exact global-batch staging therefore needs the accumulated global probe direction before the shifted evaluation.

A direct replicated-data-parallel implementation combines two original-point gradient-sized quantities and one accumulated staged quantity, versus one aggregate quantity for baseline Adam. That is approximately three times the reduction payload in this implementation, not a universal communication-time multiplier or lower bound. Parameter-sharded training can also require additional parameter gathers for the extra model traversals. Communication overlap and batching determine the actual slowdown.

Finally, the perturbation must be visible at the precision used for model computation. Reusing the same examples and controlling stochastic-layer randomness are necessary to distinguish a parameter-induced gradient change from unrelated noise. These implementation requirements remain to be tested in a language-model trainer.

## Appendix G. Synthetic checks and their limits

The companion reference implementation evaluates the staged gradients directly; Hessians are used only in independent validation calculations. Rerunning it reproduces the following checks.

For quadratic component losses, the maximum coordinate error in the exact staged-gradient identity was $3.51\times10^{-16}$. For the smooth nonlinear example, the stage remainder scaled as $O(h^2)$ and the common-state parameter remainder as $O(h^3)$. The normalization ablation with $\beta_1=\beta_2=0$ and $\epsilon=0$ reproduced the cancellation described in Appendix B.

The trajectory check used a smooth five-dimensional example, $(\beta_1,\beta_2)=(0.5,0.8)$, $\kappa=1$, $\epsilon=0$, and horizon $T=0.08$. The moments were initialized consistently through first order to remove the leading initialization effect, and no gradient coordinate crossed zero.

| Step size $h$ | Error against the modified flow with the alignment term | Error with that term omitted |
|:---|---:|---:|
| $0.002$ | $1.54\times10^{-5}$ | $4.59\times10^{-4}$ |
| $0.001$ | $3.87\times10^{-6}$ | $2.32\times10^{-4}$ |
| $0.0005$ | $9.72\times10^{-7}$ | $1.17\times10^{-4}$ |
| $0.00025$ | $2.43\times10^{-7}$ | $5.87\times10^{-5}$ |

Retaining the alignment term gave observed convergence order $1.997$; omitting it gave $0.995$. These checks support the coefficient and error order under controlled assumptions. They do not establish language-model effectiveness, stability at practical learning rates, or runtime efficiency.

The project companion files are `adam_alignment/cross_staged_adam.py` and `adam_alignment/cross_staged_validation.json`. The implementation is a small numerical reference, not a GPU optimizer. In particular, its zero-stage-size setting still evaluates the stage routines and is not a suitable timing baseline for ordinary aggregate Adam.

[barrett]: https://arxiv.org/abs/2009.11162
[dherin]: https://arxiv.org/abs/2311.00235
[memory]: https://arxiv.org/abs/2502.02132
[adam-bias]: https://arxiv.org/abs/2309.00079
[adam-ode]: https://arxiv.org/abs/1810.02263