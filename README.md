# Flow Matching Studies

Research on likelihood-based reinforcement learning for flow-matching vision-language-action models, with SmolVLA as the primary model.

The proposed study asks whether masking intermediate latent coordinates from transition log-likelihoods removes gradient terms that influence executed actions. It compares **Flow-Noise** and **Flow-SDE** under matched training and evaluation conditions.

Planned work:

- Audit which latent coordinates influence executed actions.
- Analyze masking validity and measure gradient bias and variance.
- Compare full-path and masked likelihoods in simulation using LIBERO.

**Scope:** simulation only, with a planned budget of two NVIDIA L40 GPUs and an ICRA workshop submission target.

**Status:** proposal stage. SmolVLA training and evaluation experiments have not yet been run.

See the [research proposal](proposal.md) for the mathematical formulation, related work, experimental design, and success criteria.
