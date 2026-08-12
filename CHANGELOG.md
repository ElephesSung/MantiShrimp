# Changelog

All notable changes to MantiShrimp are documented in this file.

The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-12

Initial public package release.

### Added

- Off-lattice killer-target ABM with Ornstein-Uhlenbeck motility.
- Hookean overlap repulsion and explicit killer-target adhesion.
- Continuous-time stochastic synapse binding and unbinding rules.
- Cumulative target damage, recovery, death, and killer exhaustion.
- Separate proximity-contact, bound-synapse, and attributed-kill events.
- Analysis-ready `SimulationResult` with portable CSV/JSON persistence.
- Bayesian inference for contacts or kills per killer cell using homogeneous,
  zero-inflated, Gamma-Poisson, and zero-inflated Gamma-Poisson models.
- Explicit PyMC 5 and ArviZ 0.x compatibility bounds for the initial inference
  API.
- SMC marginal-likelihood extraction, posterior summaries, and Bayes factors.
- Synthetic validation, analysis, and plotting utilities extracted from Orca.
- Typed configuration, documentation, packaging metadata, and unit tests.

[Unreleased]: https://github.com/sthsci/MantiShrimp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sthsci/MantiShrimp/releases/tag/v0.1.0
