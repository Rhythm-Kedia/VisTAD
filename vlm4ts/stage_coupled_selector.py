"""SWaT Physical Process Stage Definitions & Stage-Coupled Channel Selector.

SWaT consists of 6 sequential, physically coupled treatment stages:
- Stage 1 (Raw Water Intake & Storage): FIT101, LIT101, MV101, P101, P102
- Stage 2 (Chemical Dosing): AIT201..AIT203, FIT201, MV201, P201..P206
- Stage 3 (Ultrafiltration): DPIT301, FIT301, LIT301, MV301..MV304, P301..P302
- Stage 4 (Dechlorination): AIT401..AIT402, FIT401, LIT401, P401..P404, UV401
- Stage 5 (Reverse Osmosis): AIT501..AIT504, DPIT501, FIT501..FIT504, P501..P502, PIT501..PIT503
- Stage 6 (Clean Water Storage & Distribution): FIT601, P601..P603
"""

import numpy as np

SWAT_STAGE_CHANNELS = {
    1: {
        "name": "Stage 1: Raw Water Intake & Storage",
        "channels": [0, 1, 2, 3, 4],
        "names": ["FIT101", "LIT101", "MV101", "P101", "P102"],
        "analog": [0, 1],       # FIT101 (flow), LIT101 (level)
        "actuators": [2, 3, 4]  # MV101 (valve), P101, P102 (pumps)
    },
    2: {
        "name": "Stage 2: Chemical Dosing",
        "channels": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "names": ["AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206"],
        "analog": [5, 6, 7, 8],          # Conductivity, pH, ORP, Flow
        "actuators": [9, 10, 11, 12, 13, 14, 15] # MV201, dosing pumps P201..P206
    },
    3: {
        "name": "Stage 3: Ultrafiltration (UF)",
        "channels": [16, 17, 18, 19, 20, 21, 22, 23, 24],
        "names": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302"],
        "analog": [16, 17, 18],          # DPIT301 (diff press), FIT301 (flow), LIT301 (level)
        "actuators": [19, 20, 21, 22, 23, 24] # Valves MV301..MV304, Pumps P301..P302
    },
    4: {
        "name": "Stage 4: Dechlorination",
        "channels": [25, 26, 27, 28, 29, 30, 31, 32, 33],
        "names": ["AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403", "P404", "UV401"],
        "analog": [25, 26, 27, 28],      # Chlorine analyzers, Flow, Level
        "actuators": [29, 30, 31, 32, 33] # Pumps P401..P404, UV lamp
    },
    5: {
        "name": "Stage 5: Reverse Osmosis (RO)",
        "channels": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
        "names": ["AIT501", "AIT502", "AIT503", "AIT504", "DPIT501", "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502", "PIT501", "PIT502", "PIT503"],
        "analog": [34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46], # Analyzers, Differential Pressure, Flows, Pressures
        "actuators": [42, 43]            # High-pressure pumps P501, P502
    },
    6: {
        "name": "Stage 6: Distribution",
        "channels": [47, 48, 49, 50],
        "names": ["FIT601", "P601", "P602", "P603"],
        "analog": [47],                  # Permeate distribution flow
        "actuators": [48, 49, 50]        # Distribution pumps P601..P603
    }
}


def select_coupled_stage_channels(window_slice: np.ndarray, base_mean: np.ndarray, base_std: np.ndarray, max_channels=6):
    """Identifies the most physically perturbed stage and returns coupled channels within that stage."""
    z = np.abs((window_slice - base_mean) / np.maximum(base_std, 1e-4))
    
    stage_scores = {}
    for stage_id, meta in SWAT_STAGE_CHANNELS.items():
        chans = meta["channels"]
        # Mean absolute deviation of channels in this stage
        stage_scores[stage_id] = float(np.mean(z[:, chans]))
        
    # Select most perturbed stage
    top_stage = max(stage_scores.items(), key=lambda kv: kv[1])[0]
    meta = SWAT_STAGE_CHANNELS[top_stage]
    
    # Sort channels inside this stage by perturbation magnitude
    stage_chans = meta["channels"]
    chan_devs = [float(np.mean(z[:, c])) for c in stage_chans]
    sorted_stage_chans = [c for _, c in sorted(zip(chan_devs, stage_chans), reverse=True)]
    
    selected = sorted_stage_chans[:max_channels]
    return sorted(selected), top_stage
