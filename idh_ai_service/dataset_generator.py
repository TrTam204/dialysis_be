"""
Clinical Time-Series Vital Signs Dataset Generator for Hemodialysis IDH Modeling.
Generates realistic physiological hemodialysis vital sign trajectories, labels intradialytic
hypotension (IDH) according to KDOQI/EBPG clinical consensus, and provides train/val/test splits.
"""
from typing import Dict, List, Tuple
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    'systolic_bp',
    'diastolic_bp',
    'map',
    'heart_rate',
    'uf_rate',
    'cumulative_uf',
    'time_elapsed_min',
]


def generate_hemodialysis_sessions(
    n_sessions: int = 1500,
    idh_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate hemodialysis session time-series.
    Each session has 9 time steps (every 30 minutes from 0 to 240 min).
    Returns:
        all_sequences: np.ndarray of shape (total_samples, seq_len, n_features)
        all_labels: np.ndarray of shape (total_samples,)
    """
    np.random.seed(seed)
    seq_len = 5
    n_features = len(FEATURE_NAMES)

    time_steps = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240])
    n_steps = len(time_steps)

    sequences_list = []
    labels_list = []

    for _ in range(n_sessions):
        is_idh_session = np.random.rand() < idh_ratio

        # Baseline patient parameters
        base_sbp = np.random.normal(135.0, 12.0)
        base_dbp = np.random.normal(78.0, 8.0)
        base_hr = np.random.normal(74.0, 8.0)
        target_uf_total = np.random.uniform(2.0, 4.2)  # Total UF target in Liters

        session_sbp = np.zeros(n_steps)
        session_dbp = np.zeros(n_steps)
        session_map = np.zeros(n_steps)
        session_hr = np.zeros(n_steps)
        session_ufr = np.zeros(n_steps)
        session_cum_uf = np.zeros(n_steps)

        # Baseline at t=0
        session_sbp[0] = base_sbp
        session_dbp[0] = base_dbp
        session_map[0] = base_dbp + (base_sbp - base_dbp) / 3.0
        session_hr[0] = base_hr
        session_ufr[0] = target_uf_total / 4.0  # L/hour
        session_cum_uf[0] = 0.0

        # Determine event timing if IDH session
        idh_step = np.random.randint(5, n_steps) if is_idh_session else -1

        cum_uf = 0.0
        for t in range(1, n_steps):
            delta_hours = 0.5
            hourly_ufr = target_uf_total / 4.0 + np.random.normal(0, 0.05)
            cum_uf += hourly_ufr * delta_hours
            session_cum_uf[t] = cum_uf
            session_ufr[t] = hourly_ufr

            if is_idh_session and t >= idh_step:
                # Active IDH event: SBP drops significantly, MAP drops
                drop_severity = np.random.uniform(22.0, 38.0)
                session_sbp[t] = max(70.0, session_sbp[t - 1] - drop_severity + np.random.normal(0, 2.0))
                session_dbp[t] = max(45.0, session_dbp[t - 1] - drop_severity * 0.5 + np.random.normal(0, 2.0))
                # Heart rate may drop (paradoxical vagal response) or spike
                session_hr[t] = session_hr[t - 1] + np.random.choice([-12.0, 16.0]) + np.random.normal(0, 2.0)
            elif is_idh_session and t == idh_step - 1:
                # Pre-IDH prodrome: noticeable early drop in SBP (~10-15 mmHg)
                session_sbp[t] = session_sbp[t - 1] - np.random.uniform(10.0, 16.0) + np.random.normal(0, 2.0)
                session_dbp[t] = session_dbp[t - 1] - np.random.uniform(4.0, 8.0) + np.random.normal(0, 2.0)
                session_hr[t] = session_hr[t - 1] + np.random.uniform(2.0, 8.0) + np.random.normal(0, 2.0)
            else:
                # Stable hemodialysis trajectory: gradual mild decline
                sbp_drift = np.random.normal(-1.2, 2.5)
                session_sbp[t] = max(95.0, session_sbp[t - 1] + sbp_drift)
                dbp_drift = np.random.normal(-0.6, 1.8)
                session_dbp[t] = max(55.0, session_dbp[t - 1] + dbp_drift)
                hr_drift = np.random.normal(0.2, 2.0)
                session_hr[t] = np.clip(session_hr[t - 1] + hr_drift, 55.0, 120.0)

            session_map[t] = session_dbp[t] + (session_sbp[t] - session_dbp[t]) / 3.0

        # Construct full session feature matrix of shape (9, n_features)
        session_matrix = np.column_stack([
            session_sbp,
            session_dbp,
            session_map,
            session_hr,
            session_ufr,
            session_cum_uf,
            time_steps,
        ])

        # Slice into sliding windows of length seq_len (5)
        # Window [i : i + seq_len] predicts target at step i + seq_len
        for i in range(n_steps - seq_len):
            window = session_matrix[i : i + seq_len]  # shape (5, n_features)
            target_step = i + seq_len

            # Clinical IDH definition at target step:
            # SBP < 90 mmHg OR (base_sbp - SBP_target >= 20 mmHg and MAP < 70)
            target_sbp = session_sbp[target_step]
            target_map = session_map[target_step]
            is_idh_event = int(target_sbp < 90.0 or (base_sbp - target_sbp >= 20.0 and target_map < 72.0))

            sequences_list.append(window)
            labels_list.append(is_idh_event)

    all_sequences = np.array(sequences_list, dtype=np.float32)
    all_labels = np.array(labels_list, dtype=np.float32)

    return all_sequences, all_labels


def prepare_dataset_splits(
    n_sessions: int = 1500,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Dict[str, List[float]],
]:
    """
    Generate, normalize, and split dataset into 70% Train, 15% Validation, 15% Test.
    Uses StandardScaler fitted ONLY on train data to prevent data leakage.
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_dict
    """
    X_raw, y = generate_hemodialysis_sessions(n_sessions=n_sessions, seed=seed)

    # First split: Train (70%) vs Temp (30%)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, random_state=seed, stratify=y
    )

    # Second split: Temp into Val (15%) and Test (15%)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    n_samples, seq_len, n_features = X_train_raw.shape

    # Fit scaler on 2D reshaped train features
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, n_features))

    def scale_3d(data: np.ndarray) -> np.ndarray:
        s_samples = data.shape[0]
        reshaped = data.reshape(-1, n_features)
        scaled = scaler.transform(reshaped)
        return scaled.reshape(s_samples, seq_len, n_features).astype(np.float32)

    X_train = scale_3d(X_train_raw)
    X_val = scale_3d(X_val_raw)
    X_test = scale_3d(X_test_raw)

    scaler_dict = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': FEATURE_NAMES,
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_dict
