# Agent-based modelling for killer–target cell interactions

This repository provides a lightweight **agent-based simulator** for interactions between **killer immune cells** (e.g. NK cells / T cells) and **target cells** (e.g. tumour / infected cells). The model tracks **cell migration, short-range mechanical interactions, contact formation, probabilistic killing decisions, and target death dynamics**, returning both time-resolved trajectories and analysis-friendly tables.

The core entry point is `KTSimulator()` in **`simulator.py`**. Helper functions for initial spatial seeding, forces, boundary handling, and state initialisation live in **`assistant_function.py`**. Animation utilities are in **`virsualisation.py`**.

![demo_video](./figures/vis_test.gif)

---

## Quick start

### 1) Install dependencies

You only need a standard scientific Python stack:

- `numpy`, `pandas`
- `matplotlib`
- `tqdm`
- (for notebooks) `ipython`

### 2) Run a simulation

```python
from simulator import KTSimulator

cell_history_df, killer_positions, target_positions, killP_hist, death_hist, settings = KTSimulator(
    N_KILLER=25,
    N_TARGET=100,
    SIM_DURATION=10,
    DT0=0.1,
    boundary_condition="periodic",
    region_shape="square",
    init_mode="uniform",
)
```

### 3) Visualise / animate

```python
from virsualisation import animate_KT

animate_KT(
    cell_history_df,
    killer_positions,
    target_positions,
    killer_radius=settings["KILLER_RADIUS"],
    target_radius=settings["TARGET_RADIUS"],
    box=(settings["BOX_X_MIN"], settings["BOX_X_MAX"], settings["BOX_Y_MIN"], settings["BOX_Y_MAX"]),
    boundary_condition=settings["boundary_condition"],
)
```

A ready-to-run walkthrough is provided in **`demo.ipynb`**.

---

## What the simulator returns

`KTSimulator()` returns a tuple:

1. **`cell_history_df`** (`pandas.DataFrame`): one row per cell per time step, suitable for analysis and plotting.
2. **`killer_positions`** (`np.ndarray`, shape `(steps, N_KILLER, 2)`): trajectories for killers.
3. **`target_positions`** (`np.ndarray`, shape `(steps, N_TARGET, 2)`): trajectories for targets.
4. **`killingProbability_history`** (list of arrays): per-step killing probabilities for each killer.
5. **`DeathFactor_history`** (list of arrays): per-step death-factor values for each target.
6. **`sim_settings`** (`dict`): a small dictionary of key parameters used (box, radii, boundary condition, …).

### `cell_history_df` schema

Each time step stores:

- Common:
  - `step`, `time`
  - `cell_type` in `{killer, target}`
  - `cell_id`
  - `(x, y)` position, `(mu_x, mu_y)` polarity
  - `alive_status`
  - `contacts`: list of indices of contacting partner cells
- Killers additionally:
  - `cell_state`: a scalar in `[0, 1]` controlling killing capacity
  - `killing_P`: current kill probability
- Targets additionally:
  - `Death_Factor`: accumulated death factor
  - `killed_by`: list of killer indices credited for the kill (when death is registered)

---

## Model description

The simulator is an agent-based stochastic dynamical system in 2D, with continuous-time-like updates implemented with a variable step size `dt`.

There are two populations:

- Killers: positions $\mathbf{x}^K_i(t)$, polarity $\boldsymbol\mu^K_i(t)$, internal state $s_i(t)\in[0,1]$, and killing probability $p_i(t)$.
- Targets: positions $\mathbf{x}^T_j(t)$, polarity $\boldsymbol\mu^T_j(t)$, internal state $q_j(t)\in[0,1]$, and death factor $D_j(t)$.

### 1) Initial spatial configuration

Initial cell centres are generated with a **non-overlap constraint** using either:

- **Uniform** seeding via Poisson-disk sampling (maximally packed subject to a minimum separation), or
- **Gaussian** seeding (2D normal around the box centre / region centre), with rejection to enforce minimum separation.

Supported regions:

- **Rectangle / square** (via `bounds=(x_min, x_max, y_min, y_max)`)
- **Circle** (via `centre=(x0, y0)` and `region_radius`)

A `min_distance_factor` scales the minimum permitted centre–centre separation.

### 2) Polarity dynamics (migration persistence + noise)

For each alive cell, polarity evolves as a mean-reverting stochastic process (Euler step):

$$
\boldsymbol\mu(t+dt)=\boldsymbol\mu(t) - \gamma\,\boldsymbol\mu(t)\,dt + \sigma\sqrt{dt}\,\boldsymbol\eta,
\quad \boldsymbol\eta\sim\mathcal{N}(\mathbf{0},\mathbf{I}_2)
$$

Killers and targets have separate parameters:

- `KILLER_POL_DECAY`, `KILLER_POL_NOISE`
- `TARGET_POL_DECAY`, `TARGET_POL_NOISE`

### 3) Short-range interactions: Lennard–Jones forces

Pairs of alive cells interact via a truncated Lennard–Jones force with species-dependent parameters:

- killer–killer: `LJ_EPSILON_KK`, `SIGMA_KK`
- target–target: `LJ_EPSILON_TT`, `SIGMA_TT`
- killer–target: `LJ_EPSILON_KT`, `SIGMA_KT`

The simulator uses the standard form (implemented in `assistant_function.lj_force`), applied only when $r \le 2.5\sigma$.

### 4) Position update

For each alive cell, positions update as:

$$
\mathbf{x}(t+dt)=\mathbf{x}(t)
+ M\,\boldsymbol\mu(t)\,dt
+ \sigma_{\text{trans}}\sqrt{dt}\,\boldsymbol\xi
+ \mathbf{F}(t)\,dt
$$

where:

- $M$ is `KILLER_MOTILITY` or `TARGET_MOTILITY`,
- $\sigma_{\text{trans}}$ is `KILLER_TRANS_NOISE` or `TARGET_TRANS_NOISE`,
- $\mathbf{F}(t)$ is the net Lennard–Jones force from all neighbours (summed over pairs),
- $\boldsymbol\xi\sim\mathcal{N}(\mathbf{0},\mathbf{I}_2)$.

### 5) Boundary conditions

Two boundary modes are supported:

- `periodic`: positions wrap around the box.
- `confined`: cells reflect off walls so that the *cell edge* (not centre) remains inside the box, and the corresponding polarity component flips sign (mirror reflection).

### 6) Adaptive time stepping

The simulator chooses an adaptive time step based on the maximum net force magnitude:

$$
dt = \min\left(\frac{DS}{\max_i\lVert\mathbf{F}_i\rVert},\; DT0,\; T_{\text{end}}-t\right)
$$

This stabilises motion when forces become large, while respecting the user-chosen maximum step `DT0`.

### 7) Contact detection

A killer $i$ and target $j$ are defined to be in *contact* when their separation is below a cutoff:

$$
\lVert\mathbf{x}^T_j - \mathbf{x}^K_i\rVert \le d_c
$$

The default cutoff is tied to the killer–target $\sigma$ via:

$$
d_c = 1.2\,2^{1/6}\,\sigma_{KT}
$$

Contacts are recorded each step both as:

- a boolean contact matrix $C_{ij}(t)$,
- per-cell partner lists stored in `cell_history_df["contacts"]`.

The simulator also tracks **new contacts** per step to support “probability decay per contact” modes.

### 8) Probabilistic killing decisions (per contact episode)

Each killer has a killing probability $p_i(t)$. When a killer and target enter contact, a Bernoulli decision is sampled and **memoised per (killer, target, conjugation-count)**. Concretely, the key is:

$$
(k_{\text{idx}},\; t_{\text{idx}},\; n_{\text{conj}})
$$

where $n_{\text{conj}}$ is the number of *new* contact episodes between that pair so far.

This design ensures the same pair can make a fresh decision on each new re-contact episode (rather than resampling every frame).

Two built-in probability modes:

- `kill_prob_mode="constant"`: keep $p_i$ fixed over time,
- `kill_prob_mode="decay"`: when a **new contact** occurs, the killer’s probability decreases by `DECAY_PER_CONTACT` (floored at 0).

Initial probabilities are set by `KILLINGProb_ini(...)`:

- deterministic killing: `KILL_PROBABILISTIC=False` ⇒ $p_i=1$,
- random: `KILL_PROB_INIT=None` ⇒ uniform in `[0.1, 0.9]`,
- fixed scalar: `KILL_PROB_INIT=0.3`,
- normal: `KILL_PROB_INIT=(mu, sigma)`.

### 9) Target death factor dynamics (accumulation + optional recovery)

Targets accumulate a death factor $D_j(t)$ while being “effectively killed” by contacting killers; it relaxes back toward 0 with rate `RECOVERY_SPEED` when enabled.

In the default “recovery” mode (`RECOVERY=True`), for each target $j$:

$$
\frac{dD_j}{dt} = \underbrace{k_j}_{\ge 0} - \underbrace{\rho D_j}_{\text{recovery}}
$$

The instantaneous kill rate is proportional to the target’s state $q_j$ and a mixture of a baseline and state-weighted killer contributions:

$$
k_j = q_j\left(\alpha\,n_j + (1-\alpha)\sum_{i\in\mathcal{K}_j} s_i\right)
$$

where:

- $k_j$ is the instantaneous kill-rate contribution for target $j$,
- $\mathcal{K}_j$ is the set of killers deemed to be killing target $j$ this step (from the memoised decisions),
- $n_j = |\mathcal{K}_j|$,
- $\alpha$ is set by `MIN_KILLING_RATE` vs `MAX_KILLING_RATE` in code (currently 0 and 1 respectively),
- $\rho$ is `RECOVERY_SPEED`.

A target is declared dead when its death factor reaches the threshold `KILL_DEATH_THRESHOLD`, and death is registered when the target is no longer in contact (avoids “dying mid-contact” in the default mode).

If `RECOVERY=False`, targets can be driven to threshold without recovery (instant-death mode in code is present but partially commented; the behaviour currently sets `DeathFactor` to threshold when a killing event occurs, then registers death once not in contact).

### 10) Killer state dynamics (exhaustion-like decay)

Each killer has an internal state $s_i(t)\in[0,1]$ that modulates killing effectiveness. It is initialised by `CellState_ini(...)` (supports constants, uniform, truncated normal, or mixtures of subpopulations).

During a step, the simulator accumulates:

- time spent killing per killer, $\Delta t_i$,
- time weighted by target state, $\Delta \tilde t_i = \sum_j q_j\,dt$ over targets that killer contributed to killing.

Then the state decays multiplicatively:

$$
s_i(t+dt) = \mathrm{clip}_{[0,1]}\Bigl(s_i(t) - s_i(t)\,[a\,\Delta t_i + (b-a)\,\Delta\tilde t_i] \Bigr)
$$

with `a=MIN_STATE_DECAY_dt` and `b=MAX_STATE_DECAY_dt`.

---

## Repository layout and how each script is used

### `assistant_function.py`

A utilities module containing:

- **Initial seeding**:
  - Poisson-disk sampling in rectangles and circles,
  - Gaussian sampling with hard-core rejection,
  - `generate_two_populations(...)` to produce initial killer and target positions.
- **Physics / geometry**:
  - `lj_force(...)` and `calculate_ij_forces(...)` for pairwise Lennard–Jones forces,
  - `apply_boundary(...)` for `periodic` or `confined` walls.
- **Stochastic / state initialisation**:
  - `KillingProb_ini(...)` and `DeathFactor_ini(...)`,
  - `CellState_ini(...)` for flexible state mixtures,
  - `update_cell_states(...)` (available for alternative dynamics; not currently wired into the main loop),
  - `adaptive_time_step(...)` used by the simulator.

### `simulator.py`

Defines the **main simulator**: `KTSimulator(...)`.

Key stages inside the time loop:

1. Determine alive cells from the previous `cell_history_df` slice.
2. Update polarities with decay + noise.
3. Compute LJ forces among alive cells.
4. Adapt `dt` from force magnitudes.
5. Update positions and enforce boundary condition.
6. Build contact matrix, count new contacts, memoise kill decisions.
7. Update target death factors, register deaths, and record “killed by”.
8. Update killer internal states (exhaustion-like decay).
9. Append a new per-step record block to `cell_history_df`.

If you’re extending the model, this is the file to hack on first.

### `virsualisation.py`

Provides `animate_KT(...)` to animate trajectories and alive/dead status in a notebook:

- Draws the simulation box.
- Shows cell discs with different colours for killers, alive targets, and dead targets.
- Optionally draws short recent trajectories.
- Supports both `confined` and `periodic` boundary visualisation (periodic trajectories are segmented so lines don’t jump across the box).

---

## Parameter guide (most-used knobs)

Population / geometry

- `N_KILLER`, `N_TARGET`
- `KILLER_RADIUS`, `TARGET_RADIUS`
- `BOX_X_MIN`, `BOX_X_MAX`, `BOX_Y_MIN`, `BOX_Y_MAX`
- `boundary_condition`: `"periodic"` or `"confined"`

Initial seeding

- `region_shape`: `"square"`/`"rectangle"` or `"circle"`
- `init_mode`: `"uniform"` (Poisson-disk) or `"gaussian"`
- `min_distance_factor`: increase to reduce initial crowding
- `init_cell_bound`, `ini_cell_centre`, `ini_cell_region_radius`, `init_sigma`, `ini_seed`

Motility and persistence

- `KILLER_MOTILITY`, `TARGET_MOTILITY`
- `KILLER_POL_DECAY`, `TARGET_POL_DECAY`
- `KILLER_POL_NOISE`, `TARGET_POL_NOISE`
- `KILLER_TRANS_NOISE`, `TARGET_TRANS_NOISE`

Mechanical interactions

- `LJ_EPSILON_KK`, `LJ_EPSILON_TT`, `LJ_EPSILON_KT`
  - Tip: `LJ_EPSILON_KT` large makes killer–target strongly repulsive at short distances, preventing overlap.

Killing / death dynamics

- `KILL_PROBABILISTIC`, `KILL_PROB_INIT`, `kill_prob_mode`, `DECAY_PER_CONTACT`
- `KILLER_STATE_INIT`, `MIN_STATE_DECAY_dt`, `MAX_STATE_DECAY_dt`
- `DEATH_FAC_INIT`, `TARGET_STATE_INIT`
- `KILL_DEATH_THRESHOLD`, `RECOVERY`, `RECOVERY_SPEED`

Time stepping

- `SIM_DURATION`
- `DT0` (maximum step)
- `DS` (adaptive step scale)

---

## Reproducibility tips

- Use `ini_seed` for deterministic initial spatial seeding.
- For full determinism you’ll also want to seed NumPy’s global RNG before calling `KTSimulator`, because several routines currently call `np.random` directly.

---

## Known rough edges / things you may want to adjust

- `DISAPPEAR_DELAY` is currently defined but not used.
- The `RECOVERY=False` branch contains commented alternatives; if you want strict “instant kill on decision”, you can simplify that block.
- `update_cell_states(...)` exists as a general helper for more elaborate fatigue / recovery dynamics, but the main loop currently uses an explicit decay rule.

---

## Licence

[license](./LICENSE)