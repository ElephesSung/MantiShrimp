import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('./')
from assistant_function import *


def KTSimulator(
    N_KILLER: int = 25,
    N_TARGET: int = 100,
    *,
    KILLER_RADIUS: float = 10.0,
    TARGET_RADIUS: float = 12.0,
    # Migration
    KILLER_POL_DECAY: float = 2.5,
    TARGET_POL_DECAY: float = 2.5,
    KILLER_POL_NOISE: float = 1,
    TARGET_POL_NOISE: float = 1,
    KILLER_MOTILITY: float = 150,
    TARGET_MOTILITY: float = 100,
    KILLER_TRANS_NOISE: float = 1,
    TARGET_TRANS_NOISE: float = 1,
    # LJ interaction
    LJ_EPSILON_KK: float = 1,
    LJ_EPSILON_TT: float = 1,
    LJ_EPSILON_KT: float = 1000,
    # Simulation box
    BOX_X_MIN: float = -600,
    BOX_X_MAX: float = 600,
    BOX_Y_MIN: float = -600,
    BOX_Y_MAX: float = 600,
    boundary_condition: str = 'periodic',  # 'confined' or 'periodic'
    #Initial Spatial Distribution of Cells
    region_shape: str = "square",
    init_mode: str = "uniform",
    init_sigma: float = None,
    init_cell_bound: tuple = None,
    ini_cell_centre: tuple = None,
    ini_cell_region_radius: float = None,
    ini_seed: int = None,
    min_distance_factor: float = 2.5,
    # Time
    SIM_DURATION: float = 10,
    DT0: float = 1/10,
    DS: float = 40,
    
    # Killer initialisation
    KILL_PROBABILISTIC: bool = False,
    KILL_PROB_INIT: np.ndarray = None,
    kill_prob_mode: str = 'constant',  # 'constant'or 'decay'
    DECAY_PER_CONTACT: float = 0.05, # no worries this is for the killing probability decay.
    KILLER_STATE_INIT: float = 1.0,  # can be scalar, None, (mu, sigma), or list of (ratio, state)
    
    MIN_STATE_DECAY_dt: float = 0.001,
    MAX_STATE_DECAY_dt: float = 0.1,
    
    # Target initialisation
    DEATH_FAC_INIT: np.ndarray = None,
    KILL_DEATH_THRESHOLD: float = 1.0,
    
    TARGET_STATE_INIT: float = 1.0,  # can be scalar, None, (mu, sigma), or list of (ratio, state)
    
    RECOVERY: bool = True, 
    RECOVERY_SPEED: float = 0,
    DISAPPEAR_DELAY: float = 0.5,
    
):
    
    MIN_KILLING_RATE = 0.0
    MAX_KILLING_RATE = 1.0
    '''--- Initialisation the migration for all cells (polarities and positions) ---'''
    killer_positions_input, target_positions_input = generate_two_populations(
        N1=N_KILLER, N2 = N_TARGET,
        radius1 = KILLER_RADIUS, radius2=TARGET_RADIUS,
        shape  = region_shape,
        mode   = init_mode,
        sigma  = init_sigma,
        bounds = init_cell_bound if init_cell_bound is not None else (0.9*BOX_X_MIN, 0.9*BOX_X_MAX, 0.9*BOX_Y_MIN, 0.9*BOX_Y_MAX),
        centre = ini_cell_centre,
        region_radius = ini_cell_region_radius,
        seed   = ini_seed,
        min_distance_factor = min_distance_factor
    )
    killer_positions = np.expand_dims(killer_positions_input, axis=0)
    target_positions = np.expand_dims(target_positions_input, axis=0)
    
    killer_polarity = np.zeros((1, N_KILLER, 2))
    target_polarity = np.zeros((1, N_TARGET, 2))
    
    SIGMA_KK = (2 * KILLER_RADIUS) / (2 ** (1/6))
    SIGMA_TT = (2 * TARGET_RADIUS) / (2 ** (1/6))
    SIGMA_KT = (KILLER_RADIUS + TARGET_RADIUS) / (2 ** (1/6))
    
    '''--- Initialisation the killing and death parameters---'''
    
    ## Killing Probability initialisation
    kill_init = KillingProb_ini(N_KILLER, KILL_PROB_INIT, KILL_PROBABILISTIC)
    killingProbability_history = [kill_init]
    killer_state_init = CellState_ini(N_KILLER, STATE_INIT = KILLER_STATE_INIT)
    killer_state_history = [killer_state_init]
    
    ## Death Factor initialisation
    death_init = DeathFactor_ini(N_TARGET, DEATH_FAC_INIT)
    DeathFactor_history = [death_init]
    target_sen_state_init = CellState_ini(N_TARGET, STATE_INIT = TARGET_STATE_INIT)
    target_state_history = [target_sen_state_init]
    
    ## Death status initialisation
    kill_prob_mode = kill_prob_mode  # 'constant'or 'decay'
    decay_per_contact = DECAY_PER_CONTACT     # Only relevant for 'decay' mode
    
    
    
    '''--- Time arrays for analysis and animation ---'''
    step_list, dt_list, processed_time_list = [], [], []
    step, processed_time, dt = 0, 0, DT0
    
    '''--- Main Simulation Time Loop ---'''
    pbar = tqdm(total=SIM_DURATION, desc="Simulation progress", leave=False)
    cell_history_df = pd.DataFrame()
    # Tracking variables for contact/killing
    kill_decision_map = dict()
    contact_counts = np.zeros((N_KILLER, N_TARGET), dtype=int)
    prev_contact_matrix = np.zeros((N_KILLER, N_TARGET), dtype=bool)
    current_kill_probs = killingProbability_history[0].copy()

    # Bookkeeping for cell contacts and "killed by"
    killer_contacts_this_step = [[] for _ in range(N_KILLER)]
    target_contacts_this_step = [[] for _ in range(N_TARGET)]
    target_killed_by = [[] for _ in range(N_TARGET)]
    last_killers_for_target = [None for _ in range(N_TARGET)]
    
    net_force = np.zeros((N_KILLER + N_TARGET, 2))
    while processed_time < SIM_DURATION:
        step_list.append(step)
        dt_list.append(dt)
        processed_time_list.append(processed_time)
        current_killer_states = killer_state_history[step] if step < len(killer_state_history) else killer_state_history[-1]
        current_target_states = target_state_history[step] if step < len(target_state_history) else target_state_history[-1]
        
        if step == 0:
            alive_killers = np.ones(N_KILLER, dtype=bool)
            alive_targets = np.ones(N_TARGET, dtype=bool)
            # Clear contact lists
            killer_contacts_this_step = [[] for _ in range(N_KILLER)]
            target_contacts_this_step = [[] for _ in range(N_TARGET)]
            target_killed_by = [[] for _ in range(N_TARGET)]
            
            
        else: 
            
            #Select the alived cells
            alive_killers = cell_history_df[
                (cell_history_df['step'] == step-1) &
                (cell_history_df['cell_type'] == 'killer')
                ].sort_values('cell_id')['alive_status'].values # shape: (N_KILLER, bool)
            alive_targets = cell_history_df[
                (cell_history_df['step'] == step-1) &
                (cell_history_df['cell_type'] == 'target')
                ].sort_values('cell_id')['alive_status'].values # shape: (N_TARGET, bool)
            
            # ---Polarity Update---
            killer_noise_mu = np.sqrt(dt) * np.random.randn(N_KILLER, 2) # Calculate the noise
            target_noise_mu = np.sqrt(dt) * np.random.randn(N_TARGET, 2)
            killer_polarity = np.concatenate((killer_polarity, np.zeros((1, N_KILLER, 2))), axis=0)
            target_polarity = np.concatenate((target_polarity, np.zeros((1, N_TARGET, 2))), axis=0)
            killer_polarity[step] = (
                killer_polarity[step-1] - KILLER_POL_DECAY * killer_polarity[step-1] * dt + KILLER_POL_NOISE * killer_noise_mu
            )
            target_polarity[step] = (
                target_polarity[step-1] - TARGET_POL_DECAY * target_polarity[step-1] * dt + TARGET_POL_NOISE * target_noise_mu
            )
            
            killer_pol_alive   = killer_polarity[step][alive_killers]
            target_pol_alive   = target_polarity[step][alive_targets]
            
            ## ---Position Update---
            killer_noise_r = np.sqrt(dt) * np.random.randn(N_KILLER, 2) # noise calculation
            target_noise_r = np.sqrt(dt) * np.random.randn(N_TARGET, 2)
            
            new_killer_pos  = killer_positions[step-1].copy()
            new_target_pos  = target_positions[step-1].copy()
            
            forces = calculate_ij_forces(
                N_KILLER, N_TARGET,
                killer_positions=killer_positions[step-1],
                target_positions=target_positions[step-1],
                killer_alive=alive_killers,  
                target_alive=alive_targets,  
                LJ_EPSILON_KK=LJ_EPSILON_KK, LJ_EPSILON_TT=LJ_EPSILON_TT, LJ_EPSILON_KT=LJ_EPSILON_KT,
                SIGMA_KK=SIGMA_KK, SIGMA_TT=SIGMA_TT, SIGMA_KT=SIGMA_KT,
                )
            all_forces = np.sum(forces, axis=1)
            
            N_TOTAL = N_KILLER + N_TARGET
            polarity_force = np.zeros((N_TOTAL, 2))
            polarity_force[:N_KILLER][alive_killers] = killer_pol_alive
            polarity_force[N_KILLER:][alive_targets] = target_pol_alive
            net_force = np.sum(forces, axis=1) + polarity_force
            
            dt = adaptive_time_step(
                net_force, DS, DT0, processed_time, SIM_DURATION
            )
            dt_list.append(dt)
            
            new_killer_pos[alive_killers] = (
                killer_positions[step-1][alive_killers]
                + KILLER_MOTILITY * killer_polarity[step-1][alive_killers] * dt
                + KILLER_TRANS_NOISE * killer_noise_r[alive_killers]
                + all_forces[:N_KILLER][alive_killers] * dt
                )
            new_target_pos[alive_targets] = (
                target_positions[step-1][alive_targets]
                + TARGET_MOTILITY * target_polarity[step-1][alive_targets] * dt
                + TARGET_TRANS_NOISE * target_noise_r[alive_targets]
                + all_forces[N_KILLER:][alive_targets] * dt
                )
            
            # For killers (all have same radius)
            new_killer_pos[alive_killers], killer_polarity[step][alive_killers] = apply_boundary(
                new_killer_pos[alive_killers], killer_polarity[step][alive_killers],
                BOX_X_MIN, BOX_X_MAX, BOX_Y_MIN, BOX_Y_MAX,
                boundary_condition, KILLER_RADIUS
                )
            # For targets (all have same radius)
            new_target_pos[alive_targets], target_polarity[step][alive_targets] = apply_boundary(
                new_target_pos[alive_targets], target_polarity[step][alive_targets],
                BOX_X_MIN, BOX_X_MAX, BOX_Y_MIN, BOX_Y_MAX,
                boundary_condition, TARGET_RADIUS
                )
            
            killer_positions = np.concatenate((killer_positions, new_killer_pos[np.newaxis]), axis=0)
            target_positions = np.concatenate((target_positions, new_target_pos[np.newaxis]), axis=0)
            
            
            ## ---Kill/Death Update---
            cutoff = 1.2 * (2 ** (1/6)) * SIGMA_KT
            contact_matrix = np.zeros((N_KILLER, N_TARGET), dtype=bool)
            killer_contacts_this_step = [[] for _ in range(N_KILLER)]  # <-- clear every step
            target_contacts_this_step = [[] for _ in range(N_TARGET)]
            
            
            # --- 1. Detect contacts for ALIVE cells only ---
            for k_idx in range(N_KILLER):
                if not alive_killers[k_idx]:
                    continue
                for t_idx in range(N_TARGET):
                    if not alive_targets[t_idx]:
                        continue
                    r_ij = target_positions[step, t_idx] - killer_positions[step, k_idx]
                    d = np.linalg.norm(r_ij)
                    if d <= cutoff:
                        contact_matrix[k_idx, t_idx] = True
            
            # --- 2. Record contact lists ONLY for living cells (for DataFrame) ---
            for k_idx in range(N_KILLER):
                if not alive_killers[k_idx]:
                    continue
                killer_contacts_this_step[k_idx] = list(np.where(contact_matrix[k_idx, :])[0])
            for t_idx in range(N_TARGET):
                if not alive_targets[t_idx]:
                    continue
                target_contacts_this_step[t_idx] = list(np.where(contact_matrix[:, t_idx])[0])
                
            # --- 3. Update contact history (for new contacts only) ---
            new_contacts = np.logical_and(contact_matrix, ~prev_contact_matrix)
            contact_counts += new_contacts.astype(int)
            
            # --- 4. Death Factor Update & Killing Decisions ---
            DeathFactor_history.append(DeathFactor_history[-1].copy()) 
            current_killer_states = killer_state_history[-1]
            
            time_killing_per_killer = np.zeros(N_KILLER, dtype=float)
            time_weighted_by_target_state = np.zeros(N_KILLER, dtype=float)
            
            for t_idx in range(N_TARGET):
                if not alive_targets[t_idx]:
                    continue
                contacting_killers = np.where(contact_matrix[:, t_idx])[0]
                killed_this_step = False
                killers_who_killed = []
                for k_idx in contacting_killers:
                    n_conj = contact_counts[k_idx, t_idx]
                    key = (k_idx, t_idx, n_conj)
                    if key not in kill_decision_map:
                        prob = current_kill_probs[k_idx]
                        kill_decision_map[key] = (np.random.rand() <= prob)
                        if kill_prob_mode == 'decay' and new_contacts[k_idx, t_idx]:
                            current_kill_probs[k_idx] = max(0.0, current_kill_probs[k_idx] - decay_per_contact)
                    if kill_decision_map[key]:
                        killed_this_step = True
                        killers_who_killed.append(k_idx)
                        if not RECOVERY:
                            break 
                idx = np.asarray(killers_who_killed, dtype=int)
                if idx.size > 0:
                    time_killing_per_killer[killers_who_killed] += dt
                    time_weighted_by_target_state[killers_who_killed] += float(current_target_states[t_idx]) * dt
                    last_killers_for_target[t_idx] = killers_who_killed[:] 
                
                if RECOVERY:
                    # Standard: accumulate DeathFactor
                    if killed_this_step and idx.size > 0:
                         kill_rate = current_target_states[t_idx] * (
                            MIN_KILLING_RATE * idx.size
                            + (MAX_KILLING_RATE - MIN_KILLING_RATE) * np.sum(current_killer_states[idx])
                            )
                    else:
                        kill_rate = 0.0
                    DeathFactor_history[step][t_idx] += (kill_rate - RECOVERY_SPEED * DeathFactor_history[step][t_idx]) * dt
                    
                    if (contacting_killers.size == 0) and (DeathFactor_history[step][t_idx] >= KILL_DEATH_THRESHOLD):
                        alive_targets[t_idx] = False
                        DeathFactor_history[step][t_idx] = KILL_DEATH_THRESHOLD
                        target_killed_by[t_idx] = last_killers_for_target[t_idx]
                    # if DeathFactor_history[step][t_idx] >= KILL_DEATH_THRESHOLD:
                    #     alive_targets[t_idx] = False
                    #     DeathFactor_history[step][t_idx] = KILL_DEATH_THRESHOLD
                    #     target_killed_by[t_idx] = killers_who_killed if killers_who_killed else list(contacting_killers)
                else:
                    # No recovery: instant death
                    # if killed_this_step:
                    #     alive_targets[t_idx] = False
                    #     DeathFactor_history[step][t_idx] = KILL_DEATH_THRESHOLD
                    #     target_killed_by[t_idx] = killers_who_killed if killers_who_killed else list(contacting_killers)  
                    if killed_this_step:
                        DeathFactor_history[step][t_idx] = KILL_DEATH_THRESHOLD
                    if (not contacting_killers.size) and (DeathFactor_history[step][t_idx] >= KILL_DEATH_THRESHOLD):
                        alive_targets[t_idx] = False
                        target_killed_by[t_idx] = last_killers_for_target[t_idx]      
            
            per_killer_decay = (
                MIN_STATE_DECAY_dt * time_killing_per_killer
                + (MAX_STATE_DECAY_dt - MIN_STATE_DECAY_dt) * time_weighted_by_target_state
                ) 
            next_killer_states = current_killer_states - per_killer_decay * current_killer_states
            next_killer_states = np.clip(next_killer_states, 0.0, 1.0)
            killer_state_history.append(next_killer_states)
            
            # --- 5. Zero out contact_matrix/counts for dead targets (ensures NO ghost contacts) ---
            for t_idx in range(N_TARGET):
                if not alive_targets[t_idx]:
                    contact_matrix[:, t_idx] = False
                    contact_counts[:, t_idx] = 0
                    
            # --- 6. Prepare for next step ---
            prev_contact_matrix = contact_matrix.copy()
            killingProbability_history.append(current_kill_probs.copy())
            
        # --- Collect step records for DataFrame ---
        step_records = []
        for i in range(N_KILLER):
            step_records.append({
                'step': step,
                'time': processed_time,
                'cell_type': 'killer',
                'cell_id': i,
                'x': killer_positions[step, i, 0],
                'y': killer_positions[step, i, 1],
                'mu_x': killer_polarity[step, i, 0],
                'mu_y': killer_polarity[step, i, 1],
                'cell_state':  float(current_killer_states[i]),  
                'alive_status': True,
                'killing_P': killingProbability_history[step][i] if step < len(killingProbability_history) else kill_init[i],
                'contacts': killer_contacts_this_step[i]
            })
        for j in range(N_TARGET):
            step_records.append({
                'step': step,
                'time': processed_time,
                'cell_type': 'target',
                'cell_id': j,
                'x': target_positions[step, j, 0],
                'y': target_positions[step, j, 1],
                'mu_x': target_polarity[step, j, 0],
                'mu_y': target_polarity[step, j, 1],
                'alive_status': bool(alive_targets[j]),
                'Death_Factor': DeathFactor_history[step][j],
                'contacts': target_contacts_this_step[j],
                'killed_by': target_killed_by[j] if len(target_killed_by[j]) > 0 else None
            })
        cell_history_df = pd.concat([cell_history_df, pd.DataFrame(step_records)], ignore_index=True)

        # --- Update time and step ---
        step += 1
        if step == 0:
            processed_time += DT0
        else:
            processed_time += dt
        pbar.n = min(processed_time, SIM_DURATION)
        pbar.update(0)
    pbar.close()
    sim_settings = {
        'KILLER_RADIUS': KILLER_RADIUS,
        'TARGET_RADIUS': TARGET_RADIUS,
        'BOX_X_MIN': BOX_X_MIN,
        'BOX_X_MAX': BOX_X_MAX,
        'BOX_Y_MIN': BOX_Y_MIN,
        'BOX_Y_MAX': BOX_Y_MAX,
        'boundary_condition': boundary_condition,
        # Add more settings if you want!
    }
    return cell_history_df, killer_positions, target_positions, killingProbability_history, DeathFactor_history, sim_settings


# KTSimulator()
# print("fuck")

