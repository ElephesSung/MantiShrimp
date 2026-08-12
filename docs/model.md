# Scientific model contract

This document records the rules implemented by `mantishrimp.simulation`. It is
intended to keep model changes explicit and reviewable.

## Scope and units

The simulation is off-lattice and two-dimensional. It contains motile killer
immune cells and target cells in either a periodic rectangular domain or a
reflecting confined domain. All spatial and temporal quantities use one
user-chosen, internally consistent unit system. The reference defaults follow
the micrometre/minute interpretation used in Szonja Skenderovic's thesis.

The package implements the refined Hookean model as its maintained core. The
older Lennard–Jones scripts remain only as historical material.

## State

Each living cell has a position and an Ornstein–Uhlenbeck polarity. Killer
cells additionally have a cytotoxic state in `[0, 1]` and a productive-synapse
probability. Targets additionally have susceptibility and cumulative death
factor. A killer–target pair can be in three increasingly specific states:

1. outside proximity;
2. in physical contact;
3. bound in a synapse, which may be productive or non-productive.

The event table records transitions between these states rather than counting
every simulation frame as a new interaction.

## Motility

For an alive cell with polarity `p`,

```text
dp = -gamma * p * dt + sigma_p * sqrt(dt) * Normal(0, I)
dx = speed * p * dt + force * dt + sigma_x * sqrt(dt) * Normal(0, I)
```

Killer and target populations have separate decay, noise, and speed
parameters. Periodic calculations use minimum-image displacements. Confined
boundaries reflect both position and the corresponding polarity component.

## Mechanics and binding

Overlapping cells repel with a Hookean force

```text
F_repulsion = k_pair * (r_contact - r) * unit_vector
```

using separate killer–killer, target–target, and killer–target stiffnesses.
A bound killer–target pair beyond direct overlap and within the capture radius
experiences constant Hookean-model adhesion.

An unbound pair in proximity binds with probability
`1 - exp(-k_bind * dt)`. A bound pair unbinds with probability
`1 - exp(-k_unbind * dt)` and also separates deterministically beyond the
capture radius. These rate-to-probability conversions make the stochastic
rules consistent under changes to the integration step.

## Cytotoxicity

When a synapse forms, its productive state is sampled once from the killer's
current killing probability. The reference homogeneous model sets this
probability to one. Optional variation and per-synapse decay support later
extensions without conflating binding with killing.

Each productive bound killer contributes to target damage at

```text
target_susceptibility * (
    killing_rate_min
    + (killing_rate_max - killing_rate_min) * killer_state
)
```

Target damage recovers at a first-order rate. A target dies once damage reaches
the configured threshold. The kill event is assigned to the killer with the
largest cumulative damage contribution, while all contributors remain
recoverable from synapse history. Killer state decreases with productive
synapse time according to target susceptibility and the configured exhaustion
rates.

## Integration order

Each step computes an adaptive `dt`, updates binding/unbinding, recomputes
forces, updates polarity and position, applies boundaries, integrates damage
and exhaustion, resolves death/separation, then records contact transitions.
The adaptive step limits the largest deterministic displacement while never
exceeding `max_dt` or the remaining duration.

## Inference observables

The Bayesian layer consumes event-episode counts per killer:

- contacts: `contact_started` events;
- synapses: `synapse_formed` events;
- kills: primary-attributed `target_killed` events.

Every killer is retained, including cells with zero events. The default
exposure is the simulation duration. This is a population-level count model,
not a full likelihood for trajectories or dependent contact networks; that
assumption should be tested when calibrating against experimental data.
