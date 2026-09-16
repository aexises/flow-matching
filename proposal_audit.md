# Mathematical and prior-art audit

Date: 16 September 2026. Target: `proposal.md`, Project A.

## Verdict

The original displayed gradient identities and Gaussian counterexample are mathematically correct under their stated positive-variance, fixed-context assumptions. The proposal needed more precise validity conditions and a narrower novelty claim. Those changes are now incorporated.

The general claim that omitting parameter-dependent latent decisions can lose reward gradients is established theory. Recent work also demonstrates this failure mechanism in diffusion language models. The remaining candidate contribution is an implementation-grounded SmolVLA study separating intermediate padding, temporal suffixes, and final discarded coordinates, with controlled learning experiments. Its practical relevance remains unverified.

## Mathematical findings

| Item | Finding and action |
|---|---|
| Full-path density and score gradient | Correct. Added fixed support, fixed masks and fixed sampler length; clarify that both parent and output samples are held fixed for likelihood rescoring. |
| Masked-gradient bias sign | Correct: masked minus full equals the negative omitted reward-weighted score. Zero aggregate bias can occur through cancellation. |
| Independence condition | Previously unspecified. Added conditional mean independence and its proof using the conditional zero-mean score. |
| Final-output masking | Added a missing special case: diagonal final-step coordinates discarded before reward have zero expected score contribution. This remains true with shared parameters. |
| Intermediate masking | Possible bias, not inevitable bias. Parameter dependence, downstream dependence and nonzero reward-weighted score must be distinguished. |
| Exact reduction | Added an autonomous retained-process factorization and reverse-order integration argument. A conditional density evaluated at omitted parents is generally not a retained-path marginal. |
| Gaussian counterexample | Correct for negative, positive and zero coupling. Added the exact marginal Gaussian score as a valid alternative estimator. |
| PPO | Added explicit full-path ratio. A masked ratio requires its own probabilistic justification. Unbiased on-policy scores do not imply equal clipped objectives; averaging log-probabilities also changes the ratio. |
| Dependency audit | Added variance-network dependence, fixed-noise probes and separate temporal/dimensional masks. A mean-only Jacobian can miss a scale-mediated effect. |
| Discounting | The existing chunk bootstrap exponent was correct; added discounted within-chunk reward and genuine-terminal handling. |
| ODE deployment | The proven gradient targets the stochastic policy. No theorem here transfers improvement to a different ODE sampler. |

A useful further baseline, when tractable, is the executed-action marginal score. Under regularity it equals the conditional expectation of the full-path score given the executed action. Conditioning the reward-weighted full estimator on that action preserves its mean and cannot increase its covariance when second moments exist. Arbitrary coordinate deletion is not this conditioning operation. This is a standard latent-variable/conditional-expectation argument, not a proposed novelty claim.

The general proof framework is [Schulman et al. (2015), Theorem 1](https://arxiv.org/abs/1506.05254). The revised proposal specializes it; it does not establish a new general policy-gradient theorem.

## Missing papers and overlap

These ten papers were absent from the input proposal. Dates identify first arXiv submission where specified; version inspection is stated separately. Overlap assessments below are this audit's conclusions, not claims made by those authors.

| Paper | Inspected evidence | Relevance and novelty consequence |
|---|---|---|
| [Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/html/2607.15200), 16 July 2026 | §3.2, Eq. 11 and the stated Appendix G counterexample | Closest conceptual overlap: omitted policy-dependent unmasking decisions miss improvement directions. Discrete order selection differs from continuous coordinate omission, but the broad mechanism cannot be claimed as new. |
| [StructRL](https://arxiv.org/html/2608.15139), 15 August 2026 | §§3–4, particularly last-step replay | Direct flow-VLA neighbor: intermediate-noise propagation and action-space exploration. Distinguish an exploration-policy change from a fixed-sampler likelihood-mask intervention. Last-step replay needs its own gradient interpretation; do not treat it as a proven exact marginal-policy estimator. |
| [LC-GRPO](https://arxiv.org/html/2608.05600), 6 August 2026 | Abstract, method and Theorem 1 assumptions | Direct overlap with training/inference mismatch. Langevin correction is analyzed under specific regularity and step-size conditions; this is not a universal guarantee for multimodal VLA distributions. |
| [Q-VGM](https://arxiv.org/html/2606.08015v4), June 2026; v4 inspected | §§III–IV | Critic-guided residual velocity learning, using explicitly stated late-step approximations. Relevant alternative to likelihood-based updates, rather than a coordinate-mask validity result. |
| [ScoRe-Flow](https://arxiv.org/html/2604.10962), April 2026 | Abstract and method screening; text search for masking | Distributional/exploration control for flow policies. Relevant to noise parameterization. No coordinate-mask result was located in this screening; this is not a full proof audit. |
| [Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053), July 2025 | Primary abstract | Uses a conditional-flow-loss ratio surrogate. Establishes prior flow-policy optimization context; does not justify deleting transition factors. |
| [Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models](https://arxiv.org/html/2510.09976), October 2025 | Abstract and §III-B screening | Prior VLA flow optimization and structure-aligned updates. Broad claims about structure-aware credit assignment are already occupied. Distinct from the preceding paper despite the shared FPO acronym. |
| [Reinforcing Action Policies by Prophesying / ProphRL](https://arxiv.org/html/2511.20633), November 2025 | Abstract and FA-GRPO/FlowScale screening | Prior VLA flow-step reweighting and world-model RL. Gradient balancing across denoising steps is not a new fallback contribution. |
| [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254), 2015 | Original NeurIPS paper, Theorem 1 | Foundational overlap: stochastic ancestors and downstream costs. Required attribution for the theoretical framing. |
| [Factored Policy Gradients](https://arxiv.org/abs/2102.10362), 2021 | Primary abstract and earlier causal-policy-gradient manuscript | Prior influence-structured credit assignment. Do not claim to invent causal removal of irrelevant score terms. |

Additional September screening found [IMLE-VLA](https://arxiv.org/abs/2609.10915), submitted 10 September 2026. Its abstract describes a single-step cIMLE action generator rather than likelihood-mask analysis, so it is peripheral to Project A and was not added to the proposal's main discussion.

Existing [DBPO, Appendix A-D](https://arxiv.org/html/2604.03540) was rechecked: its executed-prefix likelihood conditions on initial noise for a one-step stochastic interface. That differs from deleting intermediate factors. Its stronger prose about full-chunk likelihood being a mismatch should not be adopted as a general theorem: a properly specified augmented policy can have the correct unclipped gradient while carrying unnecessary variance.

The [LeRobot v0.6.0 source](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py) was also rechecked: `action_in_proj` is dense, and suffix attention uses one causal increment per action token. This supports the planned distinction between dimensions and time positions; it does not verify a particular checkpoint's effective dependence or the entire RL sampler's noise network.

## Recommended paper claim

“We characterize and test when likelihood coordinate masks preserve the on-policy gradient of stochastic SmolVLA samplers, separating final-output marginalization from intermediate latent omission, and measure the consequences for PPO learning under matched conditions.”

Treat this as a hypothesis-driven workshop study. Neither a toy counterexample nor finding a dense projection establishes a publishable robotics result. The first gate should measure the omitted reward-weighted score on controlled models and the actual checkpoint, alongside dependence probes. A confirmed negative result can narrow the claim, but does not guarantee acceptance.

## Numerical validation

Run `python3 checks/proposal_math.py` with NumPy. This audit ran it with NumPy 2.2.6: **54 checks passed**. The stored output is `checks/proposal_math_results.json`.

The 16-node Gauss–Hermite rule per noise dimension checks polynomial Gaussian expectations to floating-point tolerance, including 16 mean/coupling settings, exact marginal scores, terminal score removal, scale-only influence, and parameter-independent nuisance sampling. A separate finite Bernoulli example gives full versus reduced clipped objectives of 0.7 versus 1.0 despite equal unclipped expectations. These are mathematical checks, not trained-model evidence. No SmolVLA checkpoint, simulator or GPU experiment was run.

## Search and plagiarism limits

Searches covered flow/VLA likelihood masks, padded dimensions, causal/factored gradients, latent marginal likelihoods, and recent flow-policy RL, including July–September 2026 filters. Candidate claims were checked against primary arXiv papers, the original NeurIPS paper and pinned implementation source. Secondary indexes were used for discovery only. Inspection depth varies as recorded above; this was not an exhaustive proof review of every related paper.

Exact-phrase searches for the proposal title and its central intermediate-coordinate sentence did not surface an exact matching source in returned results. That is a very limited textual check. No Turnitin/iThenticate corpus, unpublished manuscripts or exhaustive full-text similarity comparison was available.

**No plagiarism clearance is claimed.** Conceptual overlap is substantial and must be cited; overlap alone does not establish plagiarism. No direct duplicate of the full proposed SmolVLA study was identified in inspected sources. Novelty remains conditional on the implementation audit and experimental evidence. Record upstream code licenses and attribution when implementation begins, and refresh this review before submission.
