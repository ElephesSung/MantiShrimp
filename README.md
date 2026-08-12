# MantiShrimp

MantiShrimp is a Python package for agent-based modelling of killer immune
cell–tumour cell interactions and Bayesian analysis of the resulting
cell-level event counts.

The package has two connected layers:

- an off-lattice, two-dimensional ABM using the refined Hookean interaction
  rules from Szonja Skenderovic's thesis;
- the four Bayesian count models from the Orca inference work, applicable to
  either contacts per killer cell or kills per killer cell.

The historical scripts and notebooks remain in the repository as provenance.
New work should use the importable package under `src/mantishrimp`.

## Install

```bash
python -m pip install -e .
python -m pip install -e '.[inference]'
python -m pip install -e '.[all]'        # inference, plotting, and tests
```

Python 3.10 or newer is supported.

## Simulate killer–target interactions

```python
from mantishrimp import SimulationConfig, simulate

config = SimulationConfig.szonja_baseline(
    n_killers=25,
    n_targets=100,
    duration=25.0,  # shortened from the 80-minute thesis baseline
    seed=42,
)
result = simulate(config)

print(result.summary())
print(result.contacts_per_cell())
print(result.kills_per_cell())
```

`SimulationResult` contains:

- `snapshots`: one row per recorded cell and time point;
- `events`: contact, synapse, and death events with killer/target identities;
- `final_cells`: the final recorded population state;
- `config`: the complete typed simulation configuration.

Contacts and synapses are deliberately different. A contact is a proximity
episode; a synapse is a stochastic bound state formed during contact. This
keeps the biological mechanism separate from the observable used for
inference.

## Infer heterogeneity from ABM results

Install the `inference` extra, then fit all four candidate models:

```python
from mantishrimp.inference import (
    evidence_table,
    infer_contacts,
    infer_kills,
)

contact_fits = infer_contacts(
    result,
    draws=1000,
    chains=2,
    random_seed=42,
)
kill_fits = infer_kills(
    result,
    draws=1000,
    chains=2,
    random_seed=43,
)

print(evidence_table(contact_fits))
print(evidence_table(kill_fits))
```

The candidate count models are:

| Name | Population interpretation | Count distribution |
| --- | --- | --- |
| `homo` | one shared active-cell rate | Poisson |
| `Z2P` | shared rate plus non-active cells | zero-inflated Poisson |
| `Dis2P` | Gamma-distributed cell rates | negative binomial |
| `hetero3` | Gamma rates plus non-active cells | zero-inflated negative binomial |

Models are fitted with PyMC Sequential Monte Carlo, so the returned
`FitResult` includes posterior samples and the log marginal likelihood used
for Bayes-factor comparison. Raw count arrays can also be fitted with
`fit_count_model` or `fit_model_suite`.

## Package map

- `config.py`: validated simulation parameters and the Szonja baseline;
- `simulation.py`: Hookean ABM engine;
- `results.py`: event-count extraction and result persistence;
- `inference.py`: Orca-derived Bayesian model suite;
- `validation.py`: synthetic Poisson/Gamma/zero-inflated generators;
- `analysis.py`: evidence, posterior, sweep, and persistence helpers;
- `plotting.py`: reusable posterior and Bayes-factor plots.

See [docs/model.md](docs/model.md) for the scientific rules and
[docs/notebook-migration.md](docs/notebook-migration.md) for the extracted
notebook-function map.

## Development

```bash
python -m pip install -e '.[all]'
pytest
```

The package is currently alpha software. Before using it for biological
claims, calibrate the time/space units, parameter priors, and observation
process against the experiment being modelled.
