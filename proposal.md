# Likelihood Masking and Policy Improvement in Flow-Based Vision-Language-Action Models

## Project proposition

This project investigates whether likelihood masking in flow-based VLA reinforcement learning removes gradient terms that matter for the actions a robot eventually executes. SmolVLA is the primary model, and Flow-Noise and Flow-SDE provide two stochastic-policy formulations for the comparison. The target is a simulation-only ICRA workshop submission supported by a mathematical characterization, a checkpoint-level dependency audit, and replicated online-RL experiments.

The central hypothesis is that **an intermediate latent coordinate can influence an executed action even if that coordinate is discarded from the final output**. Removing its transition log-likelihood can therefore bias the policy-gradient estimator. Whether this occurs to a practically important degree in SmolVLA remains an open experimental question.

The proposed contribution is a precise account of when masking is valid and when it changes optimization, together with evidence of the consequences for policy improvement. The project does not claim to introduce Flow-Noise, Flow-SDE, or RL adaptation of SmolVLA.

## Motivation and prior work

πRL adapts flow-based VLAs to online RL using Flow-Noise and Flow-SDE. ReinFlow develops the learned-noise stochastic-path construction, while Flow-GRPO develops an ODE-to-SDE route to tractable transition densities. These methods make likelihood-based optimization possible over augmented sampling decisions, without directly evaluating the marginal density of the final executed action. [πRL](https://arxiv.org/html/2510.25889v3), [ReinFlow](https://arxiv.org/abs/2505.22094), [Flow-GRPO](https://arxiv.org/abs/2505.05470).

Two inspected SmolVLA implementations restrict transition likelihoods to physical action dimensions and executed chunk positions. They motivate an audit of the assumptions behind those restrictions; their code alone does not establish harmful bias. [smolvla_flow_rl sampler](https://github.com/mturan33/smolvla_flow_rl/blob/4b8330ace1e35b72916688cee2b32c67466e33df/src/flow_sde.py), [FlowVLA-RL sampler](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/src/flowvla_rl/smolvla_sde.py).

SmolVLA's inspected implementation uses a dense input projection over action dimensions and constructs causal masks over action tokens. These properties must be distinguished: unused dimensions within a token can potentially influence its physical coordinates, while future tokens may be unable to affect an earlier executed prefix. Neither dependency should be inferred from tensor shape alone. [LeRobot v0.6.0 implementation](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py).

DBPO already discusses executed-prefix conditional likelihood for a one-step stochastic interface. Accordingly, the candidate novelty is the **validity boundary for masking intermediate variables in multi-step VLA samplers**, rather than prefix masking itself. The literature assessment was refreshed on 16 September 2026; the additional overlap below narrows the claim and must be rechecked before submission. [DBPO, Appendix A-D](https://arxiv.org/html/2604.03540).

### Additional prior art and novelty boundary (16 September 2026)

The general gradient argument is established theory: [Schulman et al., *Gradient Estimation Using Stochastic Computation Graphs* (2015), Theorem 1](https://arxiv.org/abs/1506.05254) handles stochastic ancestors of downstream costs. [Spooner et al., *Factored Policy Gradients* (2021)](https://arxiv.org/abs/2102.10362) uses influence structure for credit assignment. The identities below are applications of that theory, not proposed new general theorems.

[Raajesh et al., *Mask-Aware Policy Gradients for Diffusion Language Models* (16 July 2026), §3.2 and Appendix G](https://arxiv.org/html/2607.15200) already demonstrates missing improvement directions when a policy-dependent sampling decision is omitted from the gradient. Their variable is discrete unmasking order; ours is a continuous latent action coordinate. That distinction motivates experiments but does not make the general omitted-score principle new.

[Yang et al., *StructRL* (15 August 2026), §§3–4](https://arxiv.org/html/2608.15139) studies propagation of intermediate noise to executed actions and moves exploration to action space with last-step replay. The proposed study must distinguish noise propagation from likelihood-mask bias while holding the sampler fixed. [Guo et al., *LC-GRPO* (6 August 2026)](https://arxiv.org/html/2608.05600) addresses finite-step stochastic-training/ODE-inference mismatch; deployment mismatch alone is not a new contribution.

Other missing context includes [McAllister et al., *Flow Matching Policy Gradients* (2025)](https://arxiv.org/abs/2507.21053), [Lyu et al., *Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models* (2025)](https://arxiv.org/abs/2510.09976), [ProphRL (2025)](https://arxiv.org/abs/2511.20633), [ScoRe-Flow (2026)](https://arxiv.org/abs/2604.10962), and [Q-VGM (2026), inspected v4](https://arxiv.org/html/2606.08015v4). They cover alternative objectives, flow-step weighting, or exploration; they constrain broad claims about flow-policy RL without directly establishing this proposal's coordinate-mask result.

**Candidate contribution:** a verified validity boundary for coordinate masks in a pinned SmolVLA sampler, separating intermediate padding, temporal suffixes, and final discarded outputs, plus replicated evidence about gradient estimates and learning. No exact duplicate of that full study was identified in the sources inspected; originality is not established by search absence. See [the audit](proposal_audit.md) for source scope and limitations.

## Research questions

1. Which omitted latent variables have a causal influence on the executed action, and which of their sampling distributions depend on trainable parameters?
2. When does dropping their score-function terms preserve the expected policy gradient?
3. Does masking materially change gradient bias, variance, or PPO clipping in Flow-Noise and Flow-SDE?
4. Does the resulting bias–variance tradeoff affect deployment success or sample efficiency on SmolVLA?

## Mathematical formulation

Let $o$ be a fixed observation and instruction context. A sampler generates latent action tensors $z_k\in\mathbb{R}^{H\times D}$ for $k=0,\ldots,K$, where $H$ is the generated horizon and $D$ includes physical and padded action dimensions. The robot executes $a=P_hz_K\in\mathbb{R}^{h\times d}$, where $P_h$ is a coordinate-selection operator (not ordinary left matrix multiplication) selecting a fixed prefix of length $h\leq H$ and the $d\leq D$ physical coordinates.

For the initial theoretical analysis, assume a parameter-independent initial-noise law, positive diagonal Gaussian transition variances on a parameter-independent support, fixed $K$ and fixed masks, differentiable transition parameters, and sufficient integrability to exchange differentiation and expectation. Collect all trainable sampler parameters into $\theta$. The full stochastic-path density is

$$
q_\theta(z_{0:K}\mid o)
=q_0(z_0)\prod_{k=0}^{K-1}q_\theta(z_{k+1}\mid z_k,o).
$$

For a reward $R(P_hz_K)$ with no explicit parameter dependence, the conditional objective and its gradient are

$$
J(\theta\mid o)=\mathbb{E}_{q_\theta}[R(P_hz_K)],
$$

$$
\nabla_\theta J
=\mathbb{E}_{q_\theta}\left[
R(P_hz_K)\sum_{k=0}^{K-1}
\nabla_\theta\log q_\theta(z_{k+1}\mid z_k,o)
\right].
$$

This is a fixed-context, unclipped score-function analysis. Extending it to closed-loop RL requires the corresponding trajectory or policy-gradient formulation; approximate advantages and PPO clipping introduce additional issues that must be evaluated separately.

For diagonal transitions, partition scalar output coordinates into retained indices $M$ and omitted indices $\bar M$. Define the omitted score

$$
S_{\bar M}
=\sum_{k=0}^{K-1}\sum_{j\in\bar M}
\nabla_\theta\log q_\theta(z_{k+1,j}\mid z_k,o),
$$

where both the stored parent $z_k$ and sampled output $z_{k+1}$ are held fixed when differentiating a conditional density. Parameters are differentiated through the conditional mean and variance, not through sampled stochastic nodes. Scalar indices flatten the $H\times D$ tensor; $M$ is the same fixed coordinate set at every step in this initial formulation. If $g_M$ is the expected estimator using only retained coordinates, then

$$
g_M-\nabla_\theta J
=-\mathbb{E}_{q_\theta}[R(P_hz_K)S_{\bar M}].
$$

Thus, absence from the final command alone does not justify dropping an intermediate score. For this particular reward, unbiasedness is equivalent to the aggregate omitted reward-weighted score being zero; individual terms may cancel. A useful sufficient condition is conditional mean independence. Write $s_{k,j}=\nabla_\theta\log q_\theta(z_{k+1,j}\mid z_k,o)$. If

$$
\mathbb{E}[R\mid z_k,z_{k+1,j},o]=\mathbb{E}[R\mid z_k,o],
$$

then $\mathbb{E}[Rs_{k,j}]=0$: condition on $z_k,o$, use the displayed equality, and use $\mathbb{E}[s_{k,j}\mid z_k,o]=0$, obtained by differentiating the normalized conditional density. A transition factor independent of $\theta$ at fixed parents also has zero score, even when its sampled coordinate affects the reward. These are sufficient conditions, not necessary ones.

**Final versus intermediate coordinates.** Under the diagonal final transition and reward $R(P_hz_K)$, each discarded final coordinate is conditionally independent of the reward given $z_{K-1},o$. Its score can therefore be removed from the unclipped estimator, even with shared network parameters. The possible bias concerns earlier omitted coordinates that influence later retained variables. A one-step diagonal sampler with parameter-independent initial noise is an important negative control.

**A valid reduced process.** Write $z_k=(x_k,y_k)$ with retained $x_k$. A sufficient structural condition for exact reduction is

$$
q_\theta(x_{k+1},y_{k+1}\mid x_k,y_k,o)
=p_\theta(x_{k+1}\mid x_k,o)\,r_\theta(y_{k+1}\mid x_k,y_k,o),
$$

with normalized factors and reward depending only on $x_K$. Integrating the nuisance path in reverse order gives

$$
p_\theta(x_{0:K}\mid o)=q_0^x(x_0)\prod_{k=0}^{K-1}p_\theta(x_{k+1}\mid x_k,o).
$$

Here $q_0^x$ is the parameter-independent initial marginal. If retained transitions instead depend on discarded parents, their product evaluated at stored parents is generally not this reduced marginal density. Causal temporal attention can support the structural condition only if the entire sampler, including its noise-scale network, preserves it. Correlated transitions require an appropriate marginal/conditional factorization rather than deletion of scalar Gaussian terms.

**PPO limitation.** Let $\ell_M$ sum the retained transition log-densities. The full-path ratio is

$$
\rho_\theta(z)=\exp\!\left(\sum_{k,j}[\log q_\theta(z_{k+1,j}\mid z_k,o)-\log q_{\theta_{\mathrm{old}}}(z_{k+1,j}\mid z_k,o)]\right).
$$

The analogous $\exp(\ell_M(\theta)-\ell_M(\theta_{\mathrm{old}}))$ is a reduced-policy importance ratio only with a justified reduction. Otherwise label it a masked ratio surrogate. Summing and averaging log-densities are different objectives: averaging gives a root of the joint ratio. Even a valid reduced ratio and the full-path ratio can yield different clipped PPO objectives, because clipping is nonlinear. Unbiasedness here applies to the on-policy, unclipped estimator, not automatically to repeated PPO updates or ODE deployment.

### Analytically checkable counterexample

Consider an internal variable $U$ and an executed action $Y$:

$$
U\sim\mathcal{N}(\theta,1),\qquad
Y\mid U\sim\mathcal{N}(cU,1),\qquad R=Y.
$$

Then $\mathbb{E}[R]=c\theta$ and

$$
\nabla_\theta J=c
=\mathbb{E}[Y(U-\theta)].
$$

Dropping the score of $U$ and rescoring only $p(Y\mid U)$ with stored $U$ detached produces zero gradient. At $c=0$, masking is harmless; for nonzero $c$, it is biased. Positive, negative, and zero coupling cases are checked by [the numerical audit](checks/proposal_math.py). The exact executed-action marginal is $Y\sim\mathcal{N}(c\theta,1+c^2)$; its score $c(Y-c\theta)/(1+c^2)$ also gives gradient $c$. This illustrates correct marginalization, which differs from retaining only the conditional density with detached $U$. This demonstrates a possible failure mechanism, not an observed SmolVLA failure.

## Experimental design

### Phase 1: model and sampler dependency audit

Pin the checkpoint, LeRobot revision, simulator version, preprocessing, normalization, action horizon, and trainable parameter set. Begin with synchronous execution and a fixed execution horizon.

For representative observations and each denoising time, perturb omitted latent coordinates and measure changes in the eventual executed action. Audit padded dimensions and future temporal positions separately. Use finite differences and directional derivatives without constructing the full Jacobian. Check whether the corresponding omitted transition parameters depend on the trainable modules. Audit the mean and variance networks separately: a scale can depend on an omitted coordinate even when the mean does not. Hold future base-noise draws fixed for pathwise sensitivity probes, and supplement them with distributional and reward-weighted score diagnostics.

Use zero perturbation and causally isolated variables as negative controls. Repeat across perturbation magnitudes to separate numerical noise from meaningful dependence. A large output sensitivity is a diagnostic, not by itself proof of nonzero expected gradient bias.

**Decision gate:** proceed with masking as the main paper only if omitted, parameter-dependent variables have measurable downstream influence. If the actual computation graph excludes that influence, document the valid reduction and narrow the proposed claim.

### Phase 2: gradient analysis on controlled models

Construct small stochastic flow policies where expected-reward gradients can be checked by quadrature or controlled finite differences. Compare complete-path and masked estimators on the same sampled trajectories.

Measure gradient bias, variance, mean-squared error, and agreement with the reference direction. Include one-step and multi-step cases, zero and nonzero coupling, parameter-independent omitted dynamics, and small positive variances. Treat zero variance as a deterministic transition requiring different mathematics, rather than inserting it into a Gaussian log-density.

First compare unclipped estimators. Then introduce PPO-style clipping to distinguish bias caused by masking from changes caused by nonlinear clipping.

### Phase 3: controlled SmolVLA fine-tuning

Use one shared PPO implementation and critic. Compare a $2\times2$ design:

| Sampler | Complete-path likelihood | Masked likelihood |
|---|---|---|
| Flow-Noise | All stochastic transition coordinates | Current physical-coordinate/executed-prefix mask |
| Full-chain Flow-SDE | All stochastic transition coordinates | Same mask rule |

Within each sampler pair, preserve the rollout sampler, transition scales, initialization, reward, trainable parameters, optimizer, and interaction budget. The likelihood treatment is the intervention. Use shared batches for local gradient diagnostics; full training runs will diverge after updates and should not be described as sharing identical rollout data.

Use summed log-densities for full joint-path PPO ratios and the corresponding masked surrogates for the primary comparison, identifying any departure from released reference recipes. Separately test per-transition clipping if needed to connect the result to existing implementations. Hybrid ODE–SDE sampling is an optional extension because deterministic steps introduce additional gradient and support questions.

Add a terminal-only mask as a mathematical control in the toy and matched-batch audits; it is not an additional full training arm in the initial 24-run budget. Its on-policy gradient should agree in expectation with the full estimator under the stated assumptions, although clipped updates can differ. Separate physical-dimension and temporal-prefix masks in those audits before attributing a combined-mask effect.

Keep SFT without RL and continued SFT as controls. If a reduced-state estimator is proposed, derive its validity before adding it as a treatment. Clamping latent padding, freezing parts of the sampler, or changing attention modifies the policy and must be reported as a distinct intervention.

### Benchmarks and metrics

Use LIBERO-Object and LIBERO-Spatial initially. Add a horizon-focused extension only after the primary mechanism is established. The primary outcome is **ODE deployment success at a fixed primitive-interaction budget**; report stochastic-sampler success separately.

Secondary measures include learning-curve area, gradient disagreement, clipping fraction, downstream sensitivity, update memory, GPU-hours, and wall time. Record per-task results and all failed training runs.

Use disjoint development and final evaluation states, not just different seed labels. Fix episode caps, success checks, camera processing, and action scaling. Evaluate methods on matched initial states and control initial-noise randomness. Distinguish trained-task evaluation from unseen-task or perturbation generalization.

## Resources and reproducibility

The working hardware budget is **two L40 GPUs**, subject to a measured memory and throughput pilot. Prefer independent training seeds on separate GPUs when each run fits. Simulator CPU throughput and rendering must be profiled alongside learner speed.

The primary design requires four RL arms × two suites × three training seeds = **24 runs**. Expanding to five seeds requires 40 runs. SFT controls, tuning, and evaluation are additional costs. Choose the final interaction budget after profiling; no convergence time is assumed.

Match primitive environment interactions and tuning effort, and report compute separately. For each sampler, estimate the complete-minus-masked difference; then test whether this effect differs between samplers. Account for training-seed variability and, for claims across tasks, task-level clustering. Do not count repeated evaluation episodes as independent training replications.

Before optimization, verify that identical-policy rescoring reproduces per-sample log-densities. Also test action masks, short final chunks, early termination, precision consistency, finite values, positive scales, and gradient flow to the declared parameter set. If primitive-step discounting is used, use the in-chunk reward $\sum_{i=0}^{h_t-1}\gamma^i r_{t+i}$ and bootstrap factor $\gamma^{h_t}$, with bootstrapping disabled at genuine terminal states. Handle time-limit truncation separately.

## Expected contributions and success criteria

The intended deliverables are:

1. A SmolVLA-specific application of established gradient theory, with explicit masking-validity conditions and failure cases.
2. A reproducible dependency and gradient audit for a pinned SmolVLA implementation.
3. A controlled Flow-Noise/Flow-SDE study showing whether the identified approximation affects learning.
4. Practical guidance for retaining or removing latent likelihood terms, with measured costs and limitations.

The proposal succeeds as a substantive workshop paper if the analysis predicts a replicated experimental effect, or establishes a useful validity boundary across meaningful model/sampler conditions. A complete-path estimator need not win: a reproducible bias–variance tradeoff can also be informative.

If the dependence is negligible or the optimization effect is too small to resolve, report that result without claiming a defective algorithm. A toy counterexample alone is insufficient to establish a robotics contribution. An isolated implementation mistake should be presented as such, rather than generalized to flow-based VLAs.

## Main risks and scope limits

- **Weak practical effect:** padded coordinates may be effectively ignored by the checkpoint. The dependency audit is the first gate.
- **Variance dominates bias:** complete-path likelihoods may produce noisier updates or more clipping. Measure both rather than equating unbiasedness with better learning.
- **Confounded comparison:** changing noise scales, critic design, clipping, or trainable modules can obscure the masking effect. Hold these fixed within treatment pairs.
- **Stopping-time complications:** masks based on early termination need separate justification from a fixed-prefix argument.
- **Limited external validity:** evidence from SmolVLA and LIBERO supports claims about that setting. A second backbone would strengthen generality but is not required for the initial scope.
- **Novelty uncertainty:** neither the score-function identity nor marginalization is new. The claim depends on the specific multi-step VLA analysis and empirical consequences, with a final prior-art check before submission.

The first milestone is a checkpoint-level dependency audit and a matched-batch gradient comparison. Large-scale training begins only if these support the proposed mechanism.
