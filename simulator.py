"""
simulator.py
Core simulation logic — mirrors exactly the radar_simulator notebook.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

FEATURES = {
    'chair_duration':          {'family': 'gamma',              'unit': 'min',    'bounds': (0, 720)},
    'daytime_bed_duration':    {'family': 'gamma',              'unit': 'min',    'bounds': (0, 480)},
    'inactive_duration':       {'family': 'gamma',              'unit': 'min',    'bounds': (0, 960)},
    'active_duration':         {'family': 'gamma',              'unit': 'min',    'bounds': (0, 480)},
    'large_motion_count':      {'family': 'negative_binomial',  'unit': 'count',  'bounds': (0, 200)},
    'small_motion_count':      {'family': 'negative_binomial',  'unit': 'count',  'bounds': (0, 500)},
    'bed_occupancy_duration':  {'family': 'gamma',              'unit': 'min',    'bounds': (0, 600)},
    'sleep_window':            {'family': 'log_normal',         'unit': 'hours',  'bounds': (0, 12)},
    'small_motions_in_bed':    {'family': 'negative_binomial',  'unit': 'count',  'bounds': (0, 100)},
    'large_motions_night':     {'family': 'negative_binomial',  'unit': 'count',  'bounds': (0, 50)},
    'bed_exit_events':         {'family': 'negative_binomial',  'unit': 'count',  'bounds': (0, 15)},
    'sleep_fragmentation_idx': {'family': 'beta',               'unit': 'index',  'bounds': (0, 1)},
    'breathing_rate_mean':     {'family': 'normal',             'unit': 'br/min', 'bounds': (6, 30)},
    'breathing_rate_var':      {'family': 'gamma',              'unit': 'br/min', 'bounds': (0, 10)},
    'apnea_count':             {'family': 'zero_inflated_nb',   'unit': 'count',  'bounds': (0, 100)},
    'apnea_duration':          {'family': 'zero_inflated_gamma','unit': 'min',    'bounds': (0, 300)},
    'cheyne_stokes_duration':  {'family': 'zero_inflated_gamma','unit': 'min',    'bounds': (0, 240)},
    'night_bathroom_visits':   {'family': 'negative_binomial',  'unit': 'count',  'bounds': (0, 15)},
}

DISEASE_PARAMS = {

    'reduced_mobility': {
        'active_duration':        {'direction': -1, 'confounded': True,  'delta_mild': -15, 'delta_severe': -40},
        'large_motion_count':     {'direction': -1, 'confounded': True,  'delta_mild': -5,  'delta_severe': -15},
        'inactive_duration':      {'direction': +1, 'confounded': True,  'delta_mild': +20, 'delta_severe': +60},
        'chair_duration':         {'direction': +1, 'confounded': True,  'delta_mild': +15, 'delta_severe': +45},
        'daytime_bed_duration':   {'direction': +1, 'confounded': True,  'delta_mild': +10, 'delta_severe': +30},
        'breathing_rate_mean':    {'direction':  0, 'confounded': False, 'delta_mild':   0, 'delta_severe':   0},
        'cheyne_stokes_duration': {'direction':  0, 'confounded': False, 'delta_mild':   0, 'delta_severe':   0},
    },

    'heart_failure': {
        'cheyne_stokes_duration':  {'direction': +1, 'confounded': False, 'delta_mild': +5,   'delta_severe': +25},
        'night_bathroom_visits':   {'direction': +1, 'confounded': True,  'delta_mild': +2,   'delta_severe': +6},
        'bed_exit_events':         {'direction': +1, 'confounded': True,  'delta_mild': +2,   'delta_severe': +5},
        'sleep_fragmentation_idx': {'direction': +1, 'confounded': True,  'delta_mild': +0.1, 'delta_severe': +0.3},
        'breathing_rate_mean':     {'direction': +1, 'confounded': True,  'delta_mild': +1.5, 'delta_severe': +4.0},
        'active_duration':         {'direction': -1, 'confounded': True,  'delta_mild': -15,  'delta_severe': -40},
    },

    'COPD': {
        'active_duration':         {'direction': -1, 'confounded': True,  'delta_mild': -15,  'delta_severe': -40},
        'large_motion_count':      {'direction': -1, 'confounded': True,  'delta_mild': -5,   'delta_severe': -15},
        'sleep_fragmentation_idx': {'direction': +1, 'confounded': True,  'delta_mild': +0.1, 'delta_severe': +0.3},
        'small_motions_in_bed':    {'direction': +1, 'confounded': True,  'delta_mild': +3,   'delta_severe': +10},
        'breathing_rate_mean':     {'direction': +1, 'confounded': True,  'delta_mild': +1.0, 'delta_severe': +3.5},
        'chair_duration':          {'direction': +1, 'confounded': True,  'delta_mild': +15,  'delta_severe': +45},
        'inactive_duration':       {'direction': +1, 'confounded': True,  'delta_mild': +20,  'delta_severe': +50},
        'daytime_bed_duration':    {'direction': +1, 'confounded': True,  'delta_mild': +10,  'delta_severe': +30},
        'cheyne_stokes_duration':  {'direction':  0, 'confounded': False, 'delta_mild':   0,  'delta_severe':   0},
    },

    'sleep_disordered_breathing': {
        'apnea_count':             {'direction': +1, 'confounded': False, 'delta_mild': +5,   'delta_severe': +20},
        'apnea_duration':          {'direction': +1, 'confounded': False, 'delta_mild': +10,  'delta_severe': +40},
        'small_motions_in_bed':    {'direction': +1, 'confounded': True,  'delta_mild': +3,   'delta_severe': +10},
        'sleep_fragmentation_idx': {'direction': +1, 'confounded': True,  'delta_mild': +0.1, 'delta_severe': +0.3},
        'night_bathroom_visits':   {'direction': +1, 'confounded': True,  'delta_mild': +2,   'delta_severe': +5},
        'bed_exit_events':         {'direction': +1, 'confounded': True,  'delta_mild': +2,   'delta_severe': +4},
        'sleep_window':            {'direction': -1, 'confounded': False, 'delta_mild': -0.5, 'delta_severe': -1.5},
        'active_duration':         {'direction': -1, 'confounded': True,  'delta_mild': -10,  'delta_severe': -25},
    },
}

@dataclass
class ResidentProfile:
    resident_id:     str
    conditions:      Dict[str, str]
    simulation_days: int = 180

POPULATION_PRIORS = {
    'chair_duration':          {'theta': 180,  'phi': 45},
    'daytime_bed_duration':    {'theta': 60,   'phi': 30},
    'inactive_duration':       {'theta': 300,  'phi': 60},
    'active_duration':         {'theta': 90,   'phi': 20},
    'large_motion_count':      {'theta': 12,   'phi': 4},
    'small_motion_count':      {'theta': 30,   'phi': 10},
    'bed_occupancy_duration':  {'theta': 480,  'phi': 60},
    'sleep_window':            {'theta': 7.5,  'phi': 0.8},
    'small_motions_in_bed':    {'theta': 8,    'phi': 3},
    'large_motions_night':     {'theta': 2,    'phi': 1},
    'bed_exit_events':         {'theta': 1,    'phi': 1},
    'sleep_fragmentation_idx': {'theta': 0.15, 'phi': 0.05},
    'breathing_rate_mean':     {'theta': 14.5, 'phi': 1.5},
    'breathing_rate_var':      {'theta': 1.2,  'phi': 0.4},
    'apnea_count':             {'theta': 0,    'phi': 0},
    'apnea_duration':          {'theta': 0,    'phi': 0},
    'cheyne_stokes_duration':  {'theta': 0,    'phi': 0},
    'night_bathroom_visits':   {'theta': 1,    'phi': 1},
}

def build_baseline():
    baseline = {}
    for feat, vals in POPULATION_PRIORS.items():
        baseline[feat] = {
            'theta':  vals['theta'],
            'phi':    vals['phi'],
            'family': FEATURES[feat]['family'] if feat in FEATURES else 'normal',
            'bounds': FEATURES[feat]['bounds'] if feat in FEATURES else (0, None),
        }
    return baseline

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def logit_inv(x):
    return 1 / (1 + np.exp(-x))

def hazard_rate(z, lambda0, alpha):
    return lambda0 * np.exp(alpha * z)

def generate_latent_trajectory(n_days, z0=0.05, B=0.97, drift=0.001,
                                lambda0=0.02, alpha=2.0, jump_size=0.4,
                                recovery=0.15, noise_sd=0.05):
    z = np.zeros(n_days)
    z[0] = z0
    in_recovery = False
    recovery_signal = 0.0
    lambda_log = []
    for d in range(1, n_days):
        lambda_d = hazard_rate(z[d-1], lambda0, alpha)
        lambda_log.append(lambda_d)

        jump = 0.0
        if np.random.poisson(lambda_d) > 0:
            jump = jump_size
            in_recovery = True
            recovery_signal = jump_size
        rec = 0.0
        if in_recovery:
            recovery_signal *= (1 - recovery)
            rec = -recovery_signal
            if recovery_signal < 0.01:
                in_recovery = False
        logit_z = B * logit(z[d-1]) + drift + jump + rec + np.random.normal(0, noise_sd)
        z[d] = logit_inv(logit_z)
    return z, lambda_log

STATE_MAP = {
    'reduced_mobility':           'M_mobility',
    'heart_failure':              'CP_cardio',
    'COPD':                       'CP_cardio',
    'sleep_disordered_breathing': 'SDB_sleep',
}

COUNT_FAMILIES = ('negative_binomial', 'zero_inflated_nb')

STATE_CONDITION_MAP = {
    'M_mobility': ['reduced_mobility'],
    'CP_cardio': ['heart_failure', 'COPD'],
    'SDB_sleep': ['sleep_disordered_breathing'],
}


def get_latent_trajectory_params(state_key, conditions):
    params = {
        'drift': 0.0,
        'lambda0': 0.02,
        'alpha': 2.0,
    }

    if state_key == 'M_mobility':
        params.update({'drift': 0.001, 'lambda0': 0.02, 'alpha': 2.0})
    elif state_key == 'CP_cardio':
        params.update({'drift': 0.002, 'lambda0': 0.03, 'alpha': 2.0})
    elif state_key == 'SDB_sleep':
        params.update({'drift': 0.0, 'lambda0': 0.01, 'alpha': 1.5})

    for condition in STATE_CONDITION_MAP.get(state_key, []):
        severity = conditions.get(condition)
        if severity is None:
            continue

        if severity == 'severe':
            params['drift'] += 0.0025
            params['lambda0'] *= 1.7
            params['alpha'] += 0.4
        else:
            params['drift'] += 0.0012
            params['lambda0'] *= 1.25
            params['alpha'] += 0.2

    return params


DAYTIME_FEATURES = (
    'chair_duration',
    'daytime_bed_duration',
    'inactive_duration',
    'active_duration',
    'large_motion_count',
    'small_motion_count',
)

NIGHTTIME_FEATURES = (
    'bed_occupancy_duration',
    'sleep_window',
    'small_motions_in_bed',
    'large_motions_night',
    'bed_exit_events',
    'sleep_fragmentation_idx',
    'night_bathroom_visits',
)

BREATHING_FEATURES = (
    'breathing_rate_mean',
    'breathing_rate_var',
    'apnea_count',
    'apnea_duration',
    'cheyne_stokes_duration',
)


def _build_base_daily_frame(traj_df):
    return pd.DataFrame({
        'day':          traj_df['day'].astype(int),
        'M_severity':   traj_df['M_mobility'].round(3),
        'CP_severity':  traj_df['CP_cardio'].round(3),
        'SDB_severity': traj_df['SDB_sleep'].round(3),
    })


def _generate_feature_block(traj_df, baseline, resident, feature_names, disease_params):
    frame = _build_base_daily_frame(traj_df)

    for feat in feature_names:
        base_vals = baseline[feat]
        theta = base_vals['theta']
        phi = base_vals['phi']
        lo, hi = base_vals['bounds']
        values = []

        for _, day in traj_df.iterrows():
            total_shift = 0.0

            for condition, severity_level in resident.conditions.items():
                if condition not in disease_params or feat not in disease_params[condition]:
                    continue

                params = disease_params[condition][feat]
                delta = params['delta_severe'] if severity_level == 'severe' else params['delta_mild']
                z_current = day[STATE_MAP.get(condition, 'M_mobility')]
                total_shift += z_current * delta

            noise = np.random.normal(0, max(phi * 0.5, 0.01))
            value = theta + total_shift + noise
            value = max(lo, value)
            if hi is not None:
                value = min(hi, value)
            if base_vals['family'] == 'beta':
                value = np.clip(value, 0, 1)

            if base_vals['family'] in COUNT_FAMILIES:
                values.append(int(max(0, round(value))))
            else:
                values.append(round(float(value), 2))

        frame[feat] = values

    return frame


def generate_daytime(traj_df, baseline, resident):
    return _generate_feature_block(traj_df, baseline, resident, DAYTIME_FEATURES, DISEASE_PARAMS)


def generate_nighttime(traj_df, baseline, resident):
    return _generate_feature_block(traj_df, baseline, resident, NIGHTTIME_FEATURES, DISEASE_PARAMS)


def generate_breathing(traj_df, baseline, resident):
    return _generate_feature_block(traj_df, baseline, resident, BREATHING_FEATURES, DISEASE_PARAMS)

def generate_daily_features(traj_df, baseline, resident, disease_params):
    daytime = generate_daytime(traj_df, baseline, resident)
    nighttime = generate_nighttime(traj_df, baseline, resident)
    breathing = generate_breathing(traj_df, baseline, resident)

    base_columns = ['day', 'M_severity', 'CP_severity', 'SDB_severity']
    return pd.concat([
        daytime,
        nighttime.drop(columns=base_columns),
        breathing.drop(columns=base_columns),
    ], axis=1)

def label_day(row, threshold=0.3):
    if row['M_severity']   > threshold: return 'mobility_decline'
    if row['CP_severity']  > threshold: return 'cardiopulmonary_event'
    if row['SDB_severity'] > threshold: return 'sdb_elevated'
    return 'normal'

def run_simulation(resident_id, conditions, simulation_days):
    # np.random.seed(42)

    resident = ResidentProfile(
        resident_id=resident_id,
        conditions=conditions,
        simulation_days=simulation_days,
    )

    baseline = build_baseline()

    m_params = get_latent_trajectory_params('M_mobility', conditions)
    cp_params = get_latent_trajectory_params('CP_cardio', conditions)
    sdb_params = get_latent_trajectory_params('SDB_sleep', conditions)

    traj_M, hazard_M = generate_latent_trajectory(simulation_days, **m_params)
    traj_CP, hazard_CP = generate_latent_trajectory(simulation_days, **cp_params)
    traj_SDB, hazard_SDB = generate_latent_trajectory(simulation_days, **sdb_params)

    traj_df = pd.DataFrame({
        'day':        range(simulation_days),
        'M_mobility': traj_M,
        'CP_cardio':  traj_CP,
        'SDB_sleep':  traj_SDB,
    })

    synthetic_df = generate_daily_features(traj_df, baseline, resident, DISEASE_PARAMS)
    synthetic_df['ground_truth_label'] = synthetic_df.apply(label_day, axis=1)

    return {
        'resident_id':     resident_id,
        'conditions':      conditions,
        'simulation_days': simulation_days,
        'trajectories': {
            'M_mobility': traj_M.tolist(),
            'CP_cardio':  traj_CP.tolist(),
            'SDB_sleep':  traj_SDB.tolist(),
        },
        'hazards': {
            'M_mobility': hazard_M,
            'CP_cardio':  hazard_CP,
            'SDB_sleep':  hazard_SDB,
        },
        'days': synthetic_df.to_dict(orient='records'),
    }
