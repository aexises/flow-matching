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

### Streaming Flow Policy: related representation, distinct RL question

[Jiang et al., *Streaming Flow Policy* (CoRL 2025), arXiv v2](https://arxiv.org/html/2505.21851v2) learns imitation policies whose flow time follows execution time. Appendix B introduces coupled action and auxiliary latent variables. Section 9 distinguishes per-time marginals from trajectory joint distributions. This is related representation-level work, not an analysis of our likelihood masks. Its deterministic transitions do not directly supply the positive-variance Gaussian conditionals assumed here. Review scope: full text, §§1–9 and Appendices A–E; inspected 23 October 2026.

**Proposed reuse (our extension, not a result of that paper):** use the auxiliary-variable construction as inspiration for a small stochastic control problem with explicitly defined positive transition variances. Sweep coupling from omitted latent variables into executed coordinates, including a zero-coupling control. Compare full-path scores, masked scores, and a tractable exact-marginal reference before clipping; then evaluate clipping separately. Merely perturbing a parameter-independent initial latent does not demonstrate omitted-score bias. The test must include parameter-dependent omitted transition factors and verify a nonzero reward-weighted omitted score. Treat this as a new diagnostic implementation requiring validation, not as an existing SFP result or a proven SmolVLA defect.

Add a low-cost execution-horizon sweep to the diagnostic phase, keeping denoising depth fixed. Measure temporal-prefix and padded-dimension masks separately, with interaction counts and chunk discounting handled consistently. If deployment latency is reported, distinguish time to first action from amortized time per action. These diagnostic extensions do not add full training arms to the initial 24-run budget.

A streaming SmolVLA action head would be a separate architecture/training intervention. It would require demonstration preprocessing, a compatible action representation, and a new RL likelihood derivation. Defer that direction until the primary masking study is complete; it cannot serve as an isolated mask ablation.

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

## Experimental methodology and execution plan

**Protocol revision: 26 October 2026.** This is a prospective protocol, not a report of completed robot experiments. Existing numerical checks validate limited analytical examples only. The streaming-policy connection motivates additional diagnostics; the primary experiment remains a fixed-sampler comparison of likelihood treatments on SmolVLA. Proposed numerical settings below are design choices, not measured resource requirements or literature-established optima. Development-only changes must be recorded before freezing the confirmatory configuration; confirmatory sample counts must not be extended in response to interim significance.

### 1. Questions, hypotheses, and units of evidence

The study separates three questions that require different evidence:

| Question | Testable hypothesis | Evidence required | Insufficient evidence |
|---|---|---|---|
| Computational dependence | Some omitted intermediate coordinates affect the distribution of executed actions. | Architecture inspection, controlled interventions, and mean/scale sensitivity tests. | A dense layer, tensor shape, or shared parameters alone. |
| Gradient validity | For some parameter-dependent omitted factors, the expected reward-weighted score is nonzero. | Exact-gradient controlled examples; on-policy estimates with uncertainty in the actual sampler. | A single-batch gradient difference or nonzero action sensitivity alone. |
| Learning consequences | Changing the likelihood treatment changes learning under a matched budget. | Replicated training runs and evaluation, with uncertainty and failures reported. | A toy counterexample or better performance at one selected checkpoint. |

The direction of the learning effect is not predetermined. Retaining more score terms can remove masking bias but increase variance, alter clipping, and worsen optimization. Stochastic-policy gradient validity also does not guarantee improvement of a different ODE deployment policy.

The units of analysis are: an independent latent trajectory in fixed-context diagnostics; an independent environment episode for return estimation; and an independently trained policy for learning comparisons. Different denoising steps, task episodes, or checkpoints from the same training run are not independent training replications.

### 2. Freeze the experimental contract before comparison

Create a versioned configuration manifest before any confirmatory run:

- Exact SmolVLA checkpoint, download revision and checksum; LeRobot and sampler Git commits; local patches; Python, PyTorch, CUDA, driver and simulator versions.
- Observation cameras, resolutions, frame history, language prompts, state representation, action order, normalization, clipping and gripper handling.
- Generated horizon $H$, physical action dimension $d$, padded dimension $D$, execution horizon $h$, integration grid and denoising depth $K$. Read these from the actual configuration; do not assume defaults.
- Full transition equations, which steps are stochastic, noise floors/ceilings, and the mean and scale parameterizations. Include endpoint handling and any terminal deterministic step.
- Trainable parameter names and counts, frozen modules, adapters if any, critic inputs/architecture, optimizer groups, learning rates, precision and gradient accumulation.
- Simulator task lists, reward definition, reset-state pools, episode limits, success predicate, discount convention and termination/truncation handling.
- Independent random streams for model initialization, task/reset selection, sampler noise, action evaluation and minibatch shuffling.

Use synchronous collection and a fixed planned execution horizon in the primary study. Disable asynchronous queues, dynamic horizons and parameter-dependent action filters unless they are explicitly modeled. Freeze normalization statistics. Use deterministic network evaluation, including disabled dropout and stochastic depth, during **both rollout collection and likelihood rescoring**; the declared sampler noise remains active. Evaluation mode does not disable autograd: enable gradients for the declared trainable parameters during rescoring. Disabling dropout only during rescoring does not reproduce a dropout-enabled behavior distribution. If network randomness is retained, explicitly include its law and realized variables in the augmented policy and validate that alternative before using it. Trainable observation features must be recomputed for the current policy; an old cached feature is valid only when its producing modules are frozen.

Use the same trainable backbone subset in the two samplers wherever possible. Flow-Noise-specific scale parameters remain part of its declared policy, so cross-sampler differences cannot be attributed solely to the mathematical source of noise. The primary causal contrast is the likelihood intervention *within* each sampler.

#### Contract status and launch requirements

No checkpoint, LeRobot revision, stochastic sampler implementation, simulator environment or trainable-module list is pinned by this document yet. The earlier LeRobot source citation is architectural evidence, not an executable experiment lock. Do not fill missing values with assumed defaults or describe the contract as complete.

| Contract artifact | Required contents before its gate |
|---|---|
| `model-lock` | Checkpoint ID, immutable revision, weight/configuration/normalization checksums; LeRobot and sampler commits, patch hash and applicable licenses. |
| `sampler-spec` | Actual mean and scale equations at every integration index; stochastic/deterministic step map; exact sampling/rescoring floors; endpoint treatment; physical/padded dimensions and generated/executed horizons. |
| `trainable-lock` | Exact parameter names, shapes, counts, optimizer groups, frozen modules and adapters; actor/critic architecture; collection/rescoring network modes and cache rules. |
| `environment-lock` | Resolved package versions, simulator/task assets, preprocessing and action transforms, reset-state split hashes, actual host/GPU/driver, environment or container identity. |
| `diagnostic-lock` | Fixed policy revision, context bank, contrasts, projection vectors, sample counts, practical thresholds, confidence procedure, seeds and compute ceiling. |

These are required records to implement and populate, not files claimed to exist. M0 may first lock one sampler for Gates A–D. A second sampler needs its own complete lock and correctness checks before comparison. Toy controls may proceed independently; checkpoint probes require the first four records, and confirmatory gradient diagnostics additionally require `diagnostic-lock`. M6 requires both sampler locks, the full four-arm configuration and the measured campaign budget.

For deterministic endpoints, explicitly choose and record either a justified mixed deterministic/stochastic estimator or a positive-variance research variant. Do not assign a Gaussian density to a deterministic transition. The variant changes sampling and must be applied identically within its FULL/BOTH pair; the released recipe remains a separately identified reference. A verbal promise to resolve endpoints later does not pass Gate A.

### 3. Define masks explicitly

Flatten each latent tensor's coordinates into $j=1,\ldots,HD$. Define a fixed binary mask $m_{k,j}$ for each stochastic transition, with $k=0,\ldots,K-1$. Let $\mathcal{D}$ select physical dimensions at all generated time positions, $\mathcal{T}$ select the planned executed prefix across all dimensions, and $\mathcal{E}=\mathcal{D}\cap\mathcal{T}$ select executed physical coordinates.

| ID | Retained score factors | Purpose |
|---|---|---|
| FULL | Every coordinate at every stochastic transition. | Reference augmented-policy score. |
| DIM | $j\in\mathcal{D}$ at every transition. | Isolate removal of padded dimensions. |
| TIME | $j\in\mathcal{T}$ at every transition. | Isolate removal of future temporal positions. |
| BOTH | $j\in\mathcal{E}$ at every transition. | Combined physical-coordinate/prefix mask. |
| FINAL | All coordinates for $k<K-1$; only $\mathcal{E}$ at the final stochastic transition, provided it directly produces the selected output. | Unclipped negative control under diagonal final transitions and the stated reward assumptions. |
| REDUCED | Factors of a separately derived normalized marginal or autonomous retained process. | Exact reference where analytically available; never assume arbitrary masking implements it. |

The primary training matrix uses FULL and BOTH. DIM, TIME, FINAL and REDUCED belong first to toy and matched-batch diagnostics. Additional full training arms require a written budget amendment. If the pinned reference implementation uses a different mask, name and document that treatment instead of calling BOTH its exact reproduction.

Masks here are likelihood masks. They do not zero sampler coordinates, alter attention, suppress noise, or change commands sent to the environment. Such changes modify the policy and are separate interventions.

### 4. Phase A — implementation and likelihood correctness

**Objective:** establish that the logged trajectory is actually sampled from the distribution being rescored.

For every transition store the conditioning context, latent parent/output, integration time, old mean and scale or sufficient data to reproduce them, old coordinatewise log-probabilities, mask metadata and policy revision. Store raw pre-clipping latent outputs; clipping or action postprocessing must not silently turn a Gaussian observation into a censored one. Parameter-independent command transforms may be absorbed into the reward/environment map for augmented-policy analysis; distinguish this from claiming a Gaussian command density.

Required checks before training:

1. Recompute old-policy log-probabilities from stored samples. Compare coordinatewise terms and total sums; identical-policy ratios must equal one within calibrated numerical tolerance.
2. Verify positive finite scales and finite samples, means, log-probabilities and gradients. Verify scale versus variance conventions and every integration-step factor.
3. Compare analytic Gaussian scores against autodifferentiation and fixed-sample finite differences in float64 toy cases. Test mean-only and scale-only trainable parameters separately.
4. Hold both sampled parent and output tensors fixed for score differentiation. Preserve gradients through trainable mean/scale networks and observation encoders.
5. Confirm that mask changes alone leave sampled trajectories, executed commands, rewards and advantages unchanged in a fixed-batch diagnostic.
6. Test $K=1$, $h=H$, $D=d$, short chunks, true termination, time-limit truncation, all-retained and all-omitted toy masks. Zero-variance transitions must be rejected by the Gaussian scorer, not made valid by a logging-only epsilon.
7. Check that the variance floor used for sampling is exactly the one used for rescoring. A changed floor changes the sampler and must be logged as such.
8. Check collection/rescoring network-mode parity and deterministic repeated mean/scale evaluations at fixed inputs. Test GAE separately on genuine termination, time-limit reset, an ordinary continuation, and a rollout-buffer cut without a reset; verify that autoreset observations never replace the final observation used for truncation bootstrapping.

Start with tolerances around $10^{-10}$ for small float64 analytical checks. Set operational tolerances for float32/mixed precision from measured rescoring error and tensor dimension; do not apply a per-coordinate tolerance blindly to a long summed log-density. Run distribution/log-probability arithmetic in adequate precision and form ratios from log-ratios.

**Gate A:** no unexplained rescoring or gradient failures. Archive failures and fixes before accepting pilot results. A sampler with deterministic endpoints must either receive a valid mixed deterministic/stochastic derivation or be explicitly replaced by a positive-variance research variant. That variant is not an exact reproduction of the released recipe.

### 5. Phase B — controlled models with known answers

#### B1. Preserve the existing scalar reference

Retain the proposal's $U\sim\mathcal{N}(\theta,1)$, $Y\mid U\sim\mathcal{N}(cU,1)$ example. Check positive, negative and zero $c$, multiple $\theta$, the full score, the masked conditional score and the exact marginal score. This verifies a failure mechanism but contains too little structure to substitute for a multi-step VLA experiment.

#### B2. Add a multi-step action/latent system

Motivated by the auxiliary-variable representation in [Streaming Flow Policy, Appendix B](https://arxiv.org/html/2505.21851v2#A2), define the following **new diagnostic model**, not a reproduction of that paper:

$$
\begin{aligned}
x_{k+1}&=\alpha x_k+c u_k+\sigma_x\epsilon^x_k,\\
u_{k+1}&=\rho u_k+\beta+\sigma_u\epsilon^u_k,
\end{aligned}
\qquad \epsilon^x_k,\epsilon^u_k\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,1).
$$

Only $x_K$ is executed/scored. Set $(x_0,u_0)$ to a parameter-independent Gaussian. Initially let $\beta$ be the only trainable parameter and use $R=x_K$; later allow $\alpha,\rho$ and log-scales to be trained and include $R=x_K^2$. Use finite $K$, strictly positive scales, and begin with $|\alpha|,|\rho|<1$ to avoid conflating masking with unstable dynamics. Fix the start mean/covariance and record all settings.

For $w_k=(x_k,u_k)^T$, define

$$
A=\begin{bmatrix}\alpha&c\\0&\rho\end{bmatrix},\quad
b=\begin{bmatrix}0\\\beta\end{bmatrix},\quad
Q=\operatorname{diag}(\sigma_x^2,\sigma_u^2).
$$

Compute exact moments recursively:

$$
\mu_{k+1}=A\mu_k+b,\qquad
\Sigma_{k+1}=A\Sigma_kA^T+Q.
$$

Then $\mathbb{E}[x_K]=\mu_{K,x}$ and $\mathbb{E}[x_K^2]=\mu_{K,x}^2+\Sigma_{K,xx}$. Differentiate these recursions for the reference gradient and independently verify with central finite differences over several step sizes. Low-dimensional quadrature provides an additional check where practical; Monte Carlo is for sampling variance, not the definition of ground truth.

An indexing check is essential: with only $\beta$ trainable, $\partial_\beta\mathbb{E}[x_1]=0$ and $\partial_\beta\mathbb{E}[x_2]=c$. The omitted latent transition needs a later step through which to influence execution. Terminal $u_K$ remains irrelevant to this reward under the diagonal transition.

Initial diagnostic grid: $K\in\{1,2,5,10\}$ and $c\in\{0,\pm0.1,\pm0.5,\pm1\}$. Begin with unit noise and then a small set of strictly positive smaller scales. Refine only when needed to resolve a mechanism; this is a lightweight synthetic grid, not a GPU training grid.

#### B3. Controls that can falsify overbroad claims

- **No coupling:** $c=0$. Omitted latent score contributions should vanish in expectation.
- **Frozen nuisance factors:** hold latent-transition parameters fixed and train only retained dynamics. A latent can influence actions without contributing an omitted trainable score.
- **Final-only deletion:** omit only discarded final output scores. The mean gradient should agree, although variance and PPO behavior can differ.
- **Scale-mediated dependence:** include a separate model with $u$ affecting the retained transition's positive scale, e.g. $\sigma_x(u)=\sigma_{\min}+\operatorname{softplus}(s+\lambda u)$. Use a quadratic reward and quadrature or another independently validated reference; the linear-Gaussian moment recursion no longer applies.
- **Temporal isolation:** build a triangular multi-token model in which later positions cannot affect earlier executed ones, then deliberately add reverse coupling. This distinguishes temporal masking from within-token dimension masking.
- **Marginal versus conditional:** where the final Gaussian marginal is known, compare its score with the full augmented-path score and the masked score. Do not identify a retained conditional with a marginal.

For each estimator report the vector bias relative to the exact reference, covariance trace, mean-squared error, gradient norm, and directional agreement when both gradients are nonzero. Use independent sampling batches and shared samples across estimators within each batch. Include uncertainty on estimated bias; zero reference gradients require absolute errors, not unstable relative errors or undefined cosine similarities.

**Gate B:** full and exact-marginal estimators agree with the exact expectation within numerical/sampling uncertainty; negative controls behave as predicted; at least one parameter-dependent coupled case shows a reproducible masked-score error. Failure of the full reference blocks interpretation of every mask result.

### 6. Phase C — checkpoint and sampler dependency audit

Construct a development observation bank covering both planned suites, multiple tasks, several episode phases, and available success/failure trajectories. A proposed starting size is 128 contexts across the two suites; enlarge if task/phase coverage is inadequate. Label the source of every context. This bank diagnoses mechanisms and is not final evaluation data.

For each context and denoising index, probe padded dimensions and future temporal positions separately. Perturb only the selected latent coordinates. Reuse future base-noise draws when comparing perturbed and unperturbed trajectories so the response is not dominated by unrelated noise.

For a normalized direction $v$ supported on omitted coordinates, estimate

$$
D_v F(z_k)\approx\frac{F(z_k+\delta v)-F(z_k-\delta v)}{2\delta},
$$

where $F$ maps the intermediate latent state to the final executed normalized action, with later base noises fixed. Use JVP/VJP checks where supported. A starting sweep is $\delta\in\{10^{-4},10^{-3},10^{-2}\}$ in normalized coordinates; interpret it against floating-point noise and typical latent scales. Report both normalized and physical action changes after the actual postprocessing.

Audit three paths: omitted coordinate to later retained mean; omitted coordinate to later retained scale; trainable parameter to the omitted transition factor. Inspect the complete computation graph, including scale heads, preprocessing, caches and attention, rather than just the action expert. Output clipping can hide a latent response, so inspect both raw and executed outputs.

Negative controls include zero perturbation, repeated identical inputs, explicit disconnected toy coordinates, and final discarded outputs. If temporal causality is structurally enforced, document the argument and use perturbations as an implementation check. Small derivatives on sampled contexts are evidence of weak local sensitivity, not a proof of independence for all states.

**Gate C:** classify each mask component as structurally justified, empirically coupled, weak/inconclusive, or unsupported because the scorer/sampler assumptions fail. If no omitted trainable factor can influence reward-relevant outputs, do not launch expensive training to hunt for the predicted bias. A useful validity-boundary study may still proceed with a revised question.

### 7. Phase D — matched-batch gradient and PPO diagnostics

#### D1. Separate exact statements from estimators

At a frozen policy revision, collect fresh on-policy stochastic paths. Recompute every mask on the same paths, using the same detached reward or advantage. Define

$$
S_m=\sum_{k,j}m_{k,j}\nabla_\theta\log q_\theta(z_{k+1,j}\mid z_k,o),
\qquad
\widehat{\Delta}_m=\frac1N\sum_{i=1}^{N}A_i\big(S_m^{(i)}-S_{\mathrm{FULL}}^{(i)}\big).
$$

For the fixed-context objective with $A_i=R_i$, this estimates the masked-minus-full expected gradient. A context-only baseline can be subtracted without changing its expectation under the score assumptions. In closed-loop PPO with approximate GAE, label the result **surrogate-gradient disagreement**, not measured bias against an exact task-return gradient.

Use two diagnostic levels:

- **Score isolation:** fixed observations and a documented parameter-independent diagnostic function of executed actions, such as a quadratic target-tracking score. This tests estimator behavior, not manipulation success.
- **Task relevance:** simulator trajectories and Monte Carlo returns/advantages under the stochastic behavior policy. Respect policy-gradient return-to-go weighting, chunk timing and episode dependence. Use the full estimator as a theoretically motivated reference under assumptions, not as a noise-free numerical oracle.

For non-differentiable task success, simulator finite differences are not a cheap exact reference. If directional parameter perturbations are used, keep them development-only and report their sampling uncertainty and policy changes explicitly.

Use development-only batches to profile cost and choose the frozen confirmatory design below. Report contributions by denoising step, mask component and parameter group, before and after optimizer gradient clipping. Full parameter vectors may be expensive: accumulate groupwise differences and fixed random projections, explicitly noting that projections cannot establish absence of all bias directions.

**Fixed-sample confirmatory diagnostic design:**

- Primary mask contrast: BOTH minus FULL at the frozen policy, default execution horizon and summed-path scoring. DIM, TIME, FINAL, stepwise and parameter-group analyses remain exploratory unless separately registered. No mask-dependent rollout or advantage changes are allowed.
- Primary scalar summaries: four unit-normalized Rademacher directions over the concatenated trainable actor parameters of each sampler. Generate them with seed `26026`, archive their values or a reproducible streamed representation and parameter ordering, and fix them before seeing confirmatory gradients. Project the *paired gradient difference*, not each estimator onto a separately chosen direction. These directions are coordinate-dependent diagnostics, not exhaustive tests of the gradient vector; groupwise and development-gradient-aligned summaries may be reported as exploratory.
- Fixed-context level: freeze the 128-context development bank and collect 32 independent replicate blocks. Each block contains one fresh latent trajectory per context, with independent noise across contexts and blocks. Compute a bank-average paired projected difference per block: 4,096 trajectories per sampler. Contexts are held fixed, so inference concerns the average conditional effect on this bank, not a population of unseen observations. Use identical samples across masks within a block.
- Task-relevance level: collect 32 independent replicate blocks, each containing one fresh episode for every selected task. Average task-balanced, within-episode return-weighted score contributions into one paired projected difference per block. With 20 tasks across the two proposed suites this is 640 episodes per sampler. Freeze task weights, return weighting and baseline before collection. Different chunks from one episode are not independent replications. Approximate-GAE analyses retain the surrogate-disagreement label.
- Uncertainty: for each scalar contrast, compute a two-sided Student-t interval from the 32 independent block means with 31 degrees of freedom. This is an approximate finite-variance inference procedure, not an exact guarantee for heavy-tailed scores. For each diagnostic level separately, reserve a family of eight primary contrasts: four directions in each of two samplers. Use Bonferroni intervals with critical quantile $t_{31,1-0.05/(2\cdot8)}$, including when only the first sampler is run initially. Report block values, dispersion and nonfinite events; if heavy tails or invalid blocks undermine the interval, label inference inconclusive rather than silently trimming data or changing methods after inspection.
- Practical resolution: for each direction, propose a threshold of 5% of the development-only RMS block-level projected FULL gradient, using the same block definition. Record both that value and its projected update under the declared optimizer state/learning rate; freeze the numerical threshold before confirmatory collection. This is a design convention, not an established meaningful-effect scale. If the reference RMS is zero, numerically unresolved or yields an irrelevant update scale, select and justify an absolute threshold on separate development data before the lock; otherwise no negligible-effect claim is allowed.
- Decision and stopping: collect exactly the locked counts and inspect inferential outcomes once at the end. An interval excluding zero supports a nonzero projected effect; an interval wholly outside the practical band supports an effect larger than the threshold; an interval wholly inside the band supports only a small effect in that specified projection and setting. Other outcomes are inconclusive. Do not infer global absence of bias from four projections. Stop early only for correctness, nonfinite values, infrastructure failure or the frozen resource ceiling, recording the incomplete result without a significance decision. Do not add samples because an interval nearly excludes zero. A larger or changed design is a new registered experiment with independent data and an updated comparison family.

The counts above are prospective caps, not evidence that proc6 can finish them in the initial slot. If profiling shows they exceed the available budget, reduce coverage or choose a different fixed design on development data *before* confirmation and record the change. Require adequate coverage and independent blocks; do not silently retain the same scientific claim after reducing the design. An effect on the diagnostic reward alone does not establish a task-relevant gradient effect.

#### D2. Probe clipping without conflating it with score bias

First evaluate the unclipped on-policy objective at $\theta=\theta_{\mathrm{old}}$. Then create several small, predeclared parameter displacements and recompute fixed-batch PPO objectives and gradients under each mask. Include the same displacements for all masks, rather than only updates preferred by one treatment.

Log summed log-ratios, ratio quantiles, fraction clipped, gradient norms, overflow/nonfinite events and effective sample size as a weight-concentration diagnostic. Only call a likelihood-ratio KL estimate a policy KL when the densities are actually normalized policies on the same space. A masked surrogate does not automatically have that interpretation.

Compare FINAL against FULL to demonstrate that equal expected unclipped gradients need not imply equal clipped objectives. Mean log-probability normalization and per-transition clipping are optional *separate* treatments; keep them out of the primary summed-path comparison.

#### D3. Execution-horizon diagnostic

Fix $H,D,K$, the checkpoint and transition schedule. Use valid distinct horizons from approximately $\{1,H/4,H/2,H\}$ after integer rounding, retaining the deployment default as well. For pure coordinate-dependence tests, reuse generated paths and change only the projection/mask. For task-return tests, recollect rollouts for each execution horizon because replanning frequency changes the policy/environment trajectory; old returns cannot simply be relabeled.

At each horizon compare DIM, TIME and BOTH against FULL, plus FINAL. This sweep is initially a diagnostic, not a Cartesian expansion of all training runs. Choose at most one additional horizon for an optional confirmatory training ablation after the mechanism is identified.

**Gate D:** either establish a task-relevant omitted-score effect with useful uncertainty, or establish a sufficiently precise validity/variance result that motivates a revised learning question. Numerical dependence alone does not pass this gate. Unresolved support or likelihood errors must be repaired first.

### 8. Phase E — controlled online fine-tuning

#### E1. Primary factorial design

| Arm | Sampler | Likelihood treatment |
|---|---|---|
| N-F | Flow-Noise | FULL |
| N-M | Flow-Noise | BOTH |
| S-F | Full-chain positive-variance Flow-SDE | FULL |
| S-M | Full-chain positive-variance Flow-SDE | BOTH |

Use one training code path, changing only the likelihood mask within each sampler pair. Preserve initial weights, initial scale settings, critic initialization, task/reset distribution, interaction budget, optimizer hyperparameters, clipping coefficient, minibatch size, update epochs and evaluation schedule. Initial exploration may differ across sampler families; report that difference rather than attributing it to masks.

Use independent critics per training run with identical architecture and initialization rules. Do not share a live critic across diverging treatment runs. During paired diagnostics, however, freeze and share advantage values to isolate the score intervention.

Each seed block starts all four arms from the same compatible pretrained checkpoint and seed mapping. Random streams provide variance reduction at initialization; once policies change, trajectories diverge. Do not claim identical training data across complete runs.

FULL and BOTH have different score dimensionality. This is part of the intervention, but it can alter gradient norms and clipping frequency. Keep the primary learning-rate schedule shared within each pair; use an equal-budget, development-only learning-rate sensitivity analysis if needed. Report any secondary treatment-specific tuning separately and do not replace the primary result with the best selectively tuned arm.

#### E2. Task organization and run count

The default is **one multitask policy per suite**, with tasks sampled uniformly unless a predeclared alternative is justified. Use LIBERO-Object and LIBERO-Spatial, three independent training seeds and the four arms: $4\times2\times3=24$ RL runs. Verify that the common SFT initialization is competent across the chosen suite before beginning RL.

This count does not describe one policy per task. If separate task policies are necessary, the count becomes $4S\sum_s T_s$, where $S$ is the seed count and $T_s$ the number of tasks in suite $s$. Explicitly revise the budget before that change; do not silently call many per-task trainings one suite run.

No-RL evaluation uses the common checkpoint and needs no RL training run. Continued SFT is a distinct training control, using only the fixed demonstration split and a declared update/compute budget; it is not interaction-matched online RL. If replicated at the same suite/seed structure it adds six runs to the three-seed design. Any required initial suite adaptation, tuning pilots and diagnostic branches are additional work.

#### E3. Rollouts, rewards and episode boundaries

Use the benchmark's documented task success signal and a fixed reward convention. Do not introduce shaping selectively to rescue one treatment. Balance task sampling and report actual primitive interactions per task.

Let $L_t$ be the number of environment steps actually executed in a chunk. Under primitive-step discounting use

$$
r_t^{\mathrm{chunk}}=\sum_{i=0}^{L_t-1}\gamma^i r_{t+i},\qquad
\delta_t=r_t^{\mathrm{chunk}}+\gamma^{L_t}(1-d_t)V(o_{t+1})-V(o_t),
$$

where $d_t$ marks genuine termination, not automatically a time-limit truncation. Here $o_{t+1}$ denotes the successor observation *before any environment reset*. At a time limit that permits bootstrapping, use its recorded final observation, never the next episode's autoreset observation. If the limit is part of the finite-horizon task objective, treat it as a terminal boundary instead and record the corresponding time-aware objective/value inputs.

Bootstrapping and trace continuation use separate masks. Let $b_t=1-d_t$ for the bootstrap convention above, and let $c_t=1$ only when the next stored decision belongs to the same episode and its advantage is available. Define

$$
\widehat A_t=\delta_t+c_t\eta_t\widehat A_{t+1},\qquad
\eta_t=\begin{cases}
\gamma^{L_t}\lambda,&\lambda\text{ per chunk decision},\\
(\gamma\lambda)^{L_t},&\lambda\text{ decayed per primitive step}.
\end{cases}
$$

The primary choice is per-decision $\lambda$; freeze its value in the configuration. The second option defines an alternative chunk-boundary trace convention, not a claim of equivalence to primitive-step GAE with intermediate value estimates. Neither convention should change between arms.

| Boundary | Bootstrap mask $b_t$ | Trace mask $c_t$ |
|---|---|---|
| Ordinary transition with next advantage available | 1 | 1 |
| Genuine termination/reset | 0 | 0 |
| Time-limit truncation/reset with continuing-task bootstrap | 1, using final pre-reset observation | 0 |
| Rollout-buffer cut without reset | 1 | 0 for a locally truncated trace, or continue only with a correctly linked next segment |

Thus a truncated episode may bootstrap while its advantage trace still stops before the next episode. Log both masks, boundary causes, final observations and $L_t$; missing final observations block the intended truncation-bootstrap calculation.

For the primary fixed-horizon mask, retain the planned prefix $h$ even if success terminates execution early; log $L_t$ for returns and discounts. A hindsight mask that removes coordinates after observed termination is a stopping-time intervention and needs separate justification. Set episode lengths/horizon handling consistently and identify administrative truncations. Do not discard short successful episodes or failed training runs.

#### E4. Tuning and budget freeze

Run a small development pilot for both samplers and both masks. Its purpose is correctness, memory, throughput and basic learnability, not selecting the winning scientific result. Register the number of pilot configurations and their resource cap before tuning. Exclude pilot seeds and reset pools from final evaluation.

After profiling, freeze the primitive-interaction budget $B$ per training run, total rollout batch size, minibatch size, optimizer epochs and evaluation checkpoints. Choose the largest common batch configuration that fits both arms of each sampler pair, using accumulation if necessary. Do not compare arms at different interaction counts solely because one is faster.

If the summed FULL ratio is unusably unstable, record that outcome. Lowering the common learning rate or update epochs is an explicit pilot revision applied to both paired arms. Switching to mean log-probabilities is a new objective, not a numerical fix that preserves the experiment.

### 9. Evaluation, uncertainty and decision rules

The primary endpoint remains the suite-balanced **ODE deployment success at the fixed final interaction budget $B$**. Stochastic-sampler success is a key secondary endpoint because it is closer to the training objective. An improvement in one deployment mode must not be described as an improvement in both.

Proposed evaluation schedule: initial checkpoint and five evenly spaced interaction checkpoints through $B$. Use a fixed, disjoint development pool during training. Run final evaluation on a held-out reset-state pool with a predeclared episode count; an initial allocation is 50 episodes per task and deployment mode, increased uniformly if pilot binomial uncertainty is too large for the desired resolution. Task repetitions improve evaluation precision, not the number of independent trained policies. Use multiple initial sampler noises where stochasticity remains, paired across arms when meaningful.

Evaluate the final checkpoint as the primary result. A best-development-checkpoint analysis may be secondary if selection uses only development data. Do not select the checkpoint on final test success. Record exact simulator states and random seeds rather than assuming different seed labels guarantee disjoint resets.

Report:

- Per-task, per-seed and suite-macro-average success; avoid allowing longer tasks or more episodes to silently reweight the suite.
- Learning-curve area on a common interaction axis, stochastic versus ODE deployment gap, and failure/termination distributions.
- Gradient disagreement, empirical variance, clipping fractions and mean/scale-network diagnostics, clearly labeled by estimator and dataset.
- Peak allocated/reserved GPU memory, CPU utilization, rollout and update times, environment steps per second, evaluation overhead, GPU-hours and elapsed wall time.
- If measuring control latency: time to first action, amortized latency per action and tail latency with the same synchronization and batching assumptions. Do not interpret throughput as reaction latency.

For each suite and seed block, compute FULL-minus-BOTH success within Flow-Noise and within Flow-SDE. Their difference is the sampler-by-mask interaction. Report paired effects and all seed values. With only three training seeds, intervals and variance estimates are fragile: treat this as a pilot-scale workshop design, not a guarantee of statistical power. Increase to five seeds if feasible before drawing strong interaction claims.

For aggregate uncertainty, preserve pairing and resample at the training-seed level; task resampling is appropriate only for an explicitly stated task-population claim and must preserve within-policy dependence. Report finite-suite results separately from generalization claims. Avoid treating hundreds of evaluation episodes as hundreds of independent policy replications. For many secondary comparisons, predeclare a multiplicity correction or label them exploratory.

Before primary training, freeze a practical success-effect band of $\pm5$ percentage points for the within-sampler FULL-minus-BOTH contrasts, or record a justified replacement based solely on development precision and scientific relevance. Freeze the final episode count from development-only precision/budget analysis before held-out evaluation; do not increase it after seeing final treatment effects. The gradient-projection bands in D1 use different units and do not establish task-success equivalence. A nonsignificant result is not equivalence. Claim a negligible effect only when the corresponding uncertainty is narrow enough relative to the frozen band; with three training seeds this may remain unresolved. Publish all scheduled outcomes, including regressions and nulls.

### 10. Resources, scheduling and execution milestones

The current execution target is **VeRLab proc6**, replacing the earlier two-48-GB-GPU assumption. Earlier setup records identify two RTX 4090 GPUs; verify actual model, per-device memory, ownership/reservation, free VRAM, driver, local scratch space and rootless Docker/CUDA/EGL operation on the day of execution. These records do not establish current availability or sampler compatibility. Treat memory as a per-device limit and do not assume that two cards provide one pooled allocation.

Begin with CPU analytical controls and one reserved GPU for one pinned sampler's correctness, dependency and gradient diagnostics. Process observations, denoising steps and probe directions sequentially; profile the derivative path as well as inference. Use smaller microbatches and accumulation where they preserve the declared estimator. A second GPU may accelerate independent work only if allocated and profiling confirms fit; do not assume that inference fit establishes actor/critic/update fit. If a valid configuration cannot fit, document that blocker and revise resources or the declared trainable configuration before continuing.

Request an initial 48-hour machine slot for environment setup, reference rescoring, checkpoint diagnostics and short training pilots. It is an allocation proposal, not a promise that every prerequisite can be finished in 48 hours. Stop the pilot when correctness blocks are found rather than spending the remaining slot on uninterpretable runs. Size CPU workers, RAM and rendering processes from observed utilization; simulator throughput may limit the benefit of extra learner compute.

Estimate total resources from measurements:

$$
G_{\mathrm{total}}=G_{\mathrm{setup/pilot}}+\sum_{r=1}^{N_{\mathrm{runs}}}g_r t_r+G_{\mathrm{controls}}+G_{\mathrm{evaluation}}+G_{\mathrm{reruns}},
$$

where $g_r$ is GPUs used by run $r$ and $t_r$ its hours, with no double counting when evaluation is already included in a run. Only if two independent single-GPU runs both fit and both GPUs are allocated does the ideal training makespan approach half their summed hours; CPU contention, unequal runtimes and downtime reduce this ideal efficiency. The 24-run factorial remains a conditional campaign design, not a booked proc6 workload. Before M6, record measured per-arm GPU-hours, peak memory, simulator/evaluation overhead and a numerical total resource ceiling. If the campaign exceeds it, publish a revised scope before training rather than dropping inconvenient arms or seeds afterward.

| Milestone | Work and deliverable | Exit criterion | Compute category |
|---|---|---|---|
| M0 — contract | Populate model/sampler/trainable/environment locks for the first sampler; define task organization and split files. | First-sampler records complete; second sampler and campaign remain explicitly pending. | CPU and setup access. |
| M1 — analytical controls | Scalar reference and new multi-step latent tests; reference-gradient and mask tables. | Gate B passes; failures understood. | CPU; no large-model training. |
| M2 — sampler correctness | Rescoring, network-mode, episode-boundary and gradient tests for the first implementation; repeat before admitting the second. | Gate A passes separately for each admitted sampler. | One GPU, small batches. |
| M3 — checkpoint audit | Mean/scale dependency maps and temporal/dimension mask classification. | Gate C classification completed. | One GPU initially. |
| M4 — gradient diagnostics | Development pilot, frozen diagnostic lock, fixed-sample paired estimates, clipping probes and exploratory horizon sweep. | Gate D passes or question is explicitly revised; incomplete counts remain inconclusive. | One GPU plus simulator CPU. |
| M5 — feasibility/tuning | Short runs for four arms; complete both sampler locks; freeze $B$, hyperparameters, success-effect band and evaluation counts. | All four arms fit; measured campaign cost fits a numerical resource ceiling. | One GPU initially; second only if allocated. |
| M6 — conditional primary campaign | 24 RL runs plus declared controls, or a prospectively documented revised design; immutable manifests and logs. | All scheduled runs accounted for, including failures. | Sequential jobs; two independent single-GPU jobs only where feasible and allocated. |
| M7 — held-out evaluation | Final checkpoints, paired test states and both deployment modes. | Evaluation integrity checks pass. | GPUs and simulator CPU. |
| M8 — analysis/write-up | Effect tables, uncertainty, mechanism plots, limitations and reproducibility package. | Claims supported by the evidence level actually obtained. | Primarily CPU. |

M1 can proceed before the large-model environment is ready. Complete M2–M4 for the first pinned sampler before expanding to the second. M3 requires a correct sampler; M6 requires successful mechanism/feasibility gates and both locked sampler pairs. No calendar completion date is assumed until M5 supplies measured throughput.

### 11. Reproducibility package and failure policy

Deliver versioned configurations, setup instructions, environment/checkpoint hashes, test commands, reset-state splits, per-run seeds, raw metrics and analysis scripts. Keep task-specific latent buffers only as long as needed for diagnostics; record storage dtype/compression so rescoring is reproducible. Avoid storing every full latent trajectory for every long training run without a storage budget.

Each run records sampler/mask IDs, parent checkpoint, trainable modules, task distribution, effective batch sizes, interactions, optimizer steps, wall time and all failures. Save resumable optimizer/critic/RNG state. Treat infrastructure interruption separately from numerical divergence. Rerun infrastructure failures with the same configuration and seed while retaining incident logs. Do not repeatedly restart diverging arms until a favorable seed appears; configuration changes create a new experiment version.

Required figures are a causal dependency schematic, toy bias/variance curves, checkpoint sensitivity summaries, per-mask gradient diagnostics, learning curves with seed dispersion, and final paired success effects. Include the negative controls prominently.

The existing `checks/proposal_math.py` covers only the earlier analytical examples. The new multi-step model, model audits and training pipeline described here remain to be implemented and tested. The plan must never cite those future checks as completed evidence.

### 12. Scope boundary for Streaming Flow Policy reuse

The borrowed motivation is the distinction between executed coordinates and auxiliary state; credit the source whenever the diagnostic is introduced. The proposed stochastic linear model and its RL tests are our planned adaptation. A deterministic flow with initial noise is not a chain of nondegenerate Gaussian transitions. Do not attach Gaussian score terms to deterministic transitions or claim that their omission proves a bug in Streaming Flow Policy.

Streaming action-head replacement, stabilization-objective changes, correlated-noise exploration, hybrid deterministic/stochastic samplers and a second backbone are optional follow-up projects. Each changes additional mechanisms or requires additional theory. The primary campaign remains the four-arm SmolVLA experiment above.


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
