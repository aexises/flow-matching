# Flow Matching Studies

Research on likelihood-based reinforcement learning for flow-matching vision-language-action models, with SmolVLA as the primary model.

The proposed study asks whether masking intermediate latent coordinates from transition log-likelihoods removes gradient terms that influence executed actions. It compares **Flow-Noise** and **Flow-SDE** under matched training and evaluation conditions.

Planned work:

- Audit which latent coordinates influence executed actions.
- Analyze masking validity and measure gradient bias and variance.
- Compare full-path and masked likelihoods in simulation using LIBERO.

**Scope:** simulation only, targeting an ICRA workshop submission. The current execution target is VeRLab proc6, beginning with CPU controls and sequential diagnostics on one allocated GPU. Hardware availability and memory fit require preflight; the full training campaign depends on measured resource costs.

**Status:** proposal stage. SmolVLA training and evaluation experiments have not yet been run.

See the [research proposal](proposal.md) for the mathematical formulation, related work, experimental design, and success criteria.
