# Likelihood Masking and Policy Improvement in Flow-Based Vision-Language-Action Models

## Project proposition

This project investigates whether likelihood masking in flow-based VLA reinforcement learning removes gradient terms that matter for the actions a robot eventually executes. SmolVLA is the primary model, and Flow-Noise and Flow-SDE provide two stochastic-policy formulations for the comparison. The target is a simulation-only ICRA workshop submission supported by a mathematical characterization, a checkpoint-level dependency audit, and replicated online-RL experiments.

The central hypothesis is that **an intermediate latent coordinate can influence an executed action even if that coordinate is discarded from the final output**. Removing its transition log-likelihood can therefore bias the policy-gradient estimator. Whether this occurs to a practically important degree in SmolVLA remains an open experimental question.

The proposed contribution is a precise account of when masking is valid and when it changes optimization, together with evidence of the consequences for policy improvement. The project does not claim to introduce Flow-Noise, Flow-SDE, or RL adaptation of SmolVLA.

## Motivation and prior work

πRL adapts flow-based VLAs to online RL using Flow-Noise and Flow-SDE. ReinFlow develops the learned-noise stochastic-path construction, while Flow-GRPO develops an ODE-to-SDE route to tractable transition densities. These methods make likelihood-based optimization possible over augmented sampling decisions, without directly evaluating the marginal density of the final executed action. [πRL](https://arxiv.org/html/2510.25889v3), [ReinFlow](https://arxiv.org/abs/2505.22094), [Flow-GRPO](https://arxiv.org/abs/2505.05470).

Two inspected SmolVLA implementations restrict transition likelihoods to physical action dimensions and executed chunk positions. They motivate an audit of the assumptions behind those restrictions; their code alone does not establish harmful bias. [smolvla_flow_rl sampler](https://github.com/mturan33/smolvla_flow_rl/blob/4b8330ace1e35b72916688cee2b32c67466e33df/src/flow_sde.py), [FlowVLA-RL sampler](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/src/flowvla_rl/smolvla_sde.py).

SmolVLA's inspected implementation uses a dense input projection over action dimensions and constructs causal masks over action tokens. These properties must be distinguished: unused dimensions within a token can potentially influence its physical coordinates, while future tokens may be unable to affect an earlier executed prefix. Neither dependency should be inferred from tensor shape alone. [LeRobot v0.6.0 implementation](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py).

DBPO already discusses executed-prefix conditional likelihood for a one-step stochastic interface. Accordingly, the candidate novelty is the **validity boundary for masking intermediate variables in multi-step VLA samplers**, rather than prefix masking itself. This positioning is based on the literature assessment completed on 14 September 2026 and must be refreshed before submission. [DBPO, Appendix A-D](https://arxiv.org/html/2604.03540).

## Research questions

1. Which omitted latent variables have a causal influence on the executed action, and which of their sampling distributions depend on trainable parameters?
2. When does dropping their score-function terms preserve the expected policy gradient?
3. Does masking materially change gradient bias, variance, or PPO clipping in Flow-Noise and Flow-SDE?
4. Does the resulting bias–variance tradeoff affect deployment success or sample efficiency on SmolVLA?

## Mathematical formulation

Let $o$ be a fixed observation and instruction context. A sampler generates latent action tensors $z_k\in\mathbb{R}^{H\times D}$ for $k=0,\ldots,K$, where $H$ is the generated horizon and $D$ includes physical and padded action dimensions. The robot executes $a=P_hz_K\in\mathbb{R}^{h\times d}$, where $P_h$ selects a fixed prefix of length $h\leq H$ and the $d\leq D$ physical coordinates.

For the initial theoretical analysis, assume a parameter-independent initial-noise law, positive diagonal Gaussian transition variances, differentiable transition parameters, and sufficient integrability to exchange differentiation and expectation. Collect all trainable sampler parameters into $\theta$. The full stochastic-path density is

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

where each stored $z_k$ is held fixed when differentiating a conditional density. If $g_M$ is the expected estimator using only retained coordinates, then

$$
g_M-\nabla_\theta J
=-\mathbb{E}_{q_\theta}[R(P_hz_K)S_{\bar M}].
$$

Thus, a coordinate being absent from the final command is not sufficient to justify dropping its score. The project will establish sufficient conditions under which the right-hand side vanishes, such as parameter-independent omitted transition factors or an appropriate conditional independence from reward. Correlated transition covariances require a separate treatment; coordinatewise log-density deletion is not generally marginalization there.

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

Dropping the score of $U$ and rescoring only $p(Y\mid U)$ with stored $U$ detached produces zero gradient. At $c=0$, masking is harmless; for nonzero $c$, it is biased. Positive, negative, and zero coupling cases have passed numerical quadrature checks in the existing project companion. This demonstrates a possible failure mechanism, not an observed SmolVLA failure.

## Experimental design

### Phase 1: model and sampler dependency audit

Pin the checkpoint, LeRobot revision, simulator version, preprocessing, normalization, action horizon, and trainable parameter set. Begin with synchronous execution and a fixed execution horizon.

For representative observations and each denoising time, perturb omitted latent coordinates and measure changes in the eventual executed action. Audit padded dimensions and future temporal positions separately. Use finite differences and directional derivatives without constructing the full Jacobian. Check whether the corresponding omitted transition parameters depend on the trainable modules.

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

Use the same joint-path PPO ratio construction for the primary cross-sampler comparison, identifying any departure from released reference recipes. Separately test per-transition clipping if needed to connect the result to existing implementations. Hybrid ODE–SDE sampling is an optional extension because deterministic steps introduce additional gradient and support questions.

Keep SFT without RL and continued SFT as controls. If a reduced-state estimator is proposed, derive its validity before adding it as a treatment. Clamping latent padding, freezing parts of the sampler, or changing attention modifies the policy and must be reported as a distinct intervention.

### Benchmarks and metrics

Use LIBERO-Object and LIBERO-Spatial initially. Add a horizon-focused extension only after the primary mechanism is established. The primary outcome is **ODE deployment success at a fixed primitive-interaction budget**; report stochastic-sampler success separately.

Secondary measures include learning-curve area, gradient disagreement, clipping fraction, downstream sensitivity, update memory, GPU-hours, and wall time. Record per-task results and all failed training runs.

Use disjoint development and final evaluation states, not just different seed labels. Fix episode caps, success checks, camera processing, and action scaling. Evaluate methods on matched initial states and control initial-noise randomness. Distinguish trained-task evaluation from unseen-task or perturbation generalization.

## Resources and reproducibility

The working hardware budget is **two L40 GPUs**, subject to a measured memory and throughput pilot. Prefer independent training seeds on separate GPUs when each run fits. Simulator CPU throughput and rendering must be profiled alongside learner speed.

The primary design requires four RL arms × two suites × three training seeds = **24 runs**. Expanding to five seeds requires 40 runs. SFT controls, tuning, and evaluation are additional costs. Choose the final interaction budget after profiling; no convergence time is assumed.

Match primitive environment interactions and tuning effort, and report compute separately. For each sampler, estimate the complete-minus-masked difference; then test whether this effect differs between samplers. Account for training-seed variability and, for claims across tasks, task-level clustering. Do not count repeated evaluation episodes as independent training replications.

Before optimization, verify that identical-policy rescoring reproduces per-sample log-densities. Also test action masks, short final chunks, early termination, precision consistency, finite values, positive scales, and gradient flow to the declared parameter set. If primitive-step discounting is used, an executed chunk of length $h_t$ requires the corresponding $\gamma^{h_t}$ bootstrap factor.

## Expected contributions and success criteria

The intended deliverables are:

1. A mathematical statement of the masking-validity conditions and explicit failure cases.
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
