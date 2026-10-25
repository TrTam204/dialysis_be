"""
Clinical Time-Series Vital Signs Dataset Generator for Hemodialysis IDH Modeling.
Generates realistic physiological hemodialysis vital sign trajectories, incorporates
dry weight, normalized ultrafiltration rate (mL/kg/h), SBP drop from baseline,
transient borderline physiological variations, and provides zero-data-leakage
session-level train/val/test splits.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    'systolic_bp',
    'diastolic_bp',
    'map',
    'sbp_drop_from_baseline',
    'heart_rate',
    'ufr_ml_kg_h',
    'cumulative_uf_pct',
    'time_elapsed_min',
]


def _simulate_single_session(
    session_id: int,
    is_idh_session: bool,
    rng: np.random.RandomState,
    seq_len: int = 3,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Simulate a single hemodialysis session (4 hours = 9 observation steps, every 30 mins)
    and extract sliding windows of length seq_len predicting IDH at the next step (t+1).
    Includes realistic borderline cases:
    - Transient physiological dip in negative sessions (~15%) with compensatory stabilization.
    - Subtle/gradual prodrome in positive sessions (~15%).
    """
    time_steps = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240], dtype=np.float32)
    n_steps = len(time_steps)

    # 1. Patient anthropometric and baseline vital signs
    dry_weight = float(rng.uniform(42.0, 82.0))  # kg
    uf_target = float(rng.uniform(1.2, 4.2))     # Liters
    base_sbp = float(np.clip(rng.normal(135.0, 12.0), 105.0, 165.0))
    base_dbp = float(np.clip(rng.normal(78.0, 8.0), 60.0, 95.0))
    base_hr = float(np.clip(rng.normal(74.0, 8.0), 60.0, 95.0))

    session_sbp = np.zeros(n_steps, dtype=np.float32)
    session_dbp = np.zeros(n_steps, dtype=np.float32)
    session_map = np.zeros(n_steps, dtype=np.float32)
    session_hr = np.zeros(n_steps, dtype=np.float32)
    session_ufr = np.zeros(n_steps, dtype=np.float32)
    session_cum_uf = np.zeros(n_steps, dtype=np.float32)
    session_sbp_drop = np.zeros(n_steps, dtype=np.float32)
    session_cum_uf_pct = np.zeros(n_steps, dtype=np.float32)

    # Step 0 baseline
    session_sbp[0] = base_sbp
    session_dbp[0] = base_dbp
    session_map[0] = base_dbp + (base_sbp - base_dbp) / 3.0
    session_hr[0] = base_hr
    hourly_ufr_target = uf_target / 4.0  # L/hour
    session_ufr[0] = (hourly_ufr_target * 1000.0) / dry_weight  # mL/kg/h
    session_cum_uf[0] = 0.0
    session_sbp_drop[0] = 0.0
    session_cum_uf_pct[0] = 0.0

    # IDH event timing and borderline pattern flags
    is_borderline_negative = (not is_idh_session) and (rng.rand() < 0.15)
    is_subtle_positive = is_idh_session and (rng.rand() < 0.15)

    idh_step = int(rng.randint(4, n_steps)) if is_idh_session else -1
    dip_step = int(rng.randint(3, 6)) if is_borderline_negative else -1

    cum_uf = 0.0
    for t in range(1, n_steps):
        delta_hours = 0.5
        hourly_ufr_liters = hourly_ufr_target + float(rng.normal(0, 0.05))
        cum_uf += hourly_ufr_liters * delta_hours
        session_cum_uf[t] = cum_uf
        session_ufr[t] = (hourly_ufr_liters * 1000.0) / dry_weight
        session_cum_uf_pct[t] = (cum_uf / dry_weight) * 100.0

        noise_s = float(rng.normal(0, 2.5))
        noise_d = float(rng.normal(0, 1.8))
        noise_h = float(rng.normal(0, 2.0))

        if is_idh_session and t >= idh_step:
            # Active IDH event: SBP drops significantly, MAP drops
            drop_severity = float(rng.uniform(22.0, 36.0))
            session_sbp[t] = max(70.0, session_sbp[t - 1] - drop_severity + noise_s)
            session_dbp[t] = max(45.0, session_dbp[t - 1] - drop_severity * 0.5 + noise_d)
            hr_shift = float(rng.choice([-10.0, 14.0])) + noise_h
            session_hr[t] = np.clip(session_hr[t - 1] + hr_shift, 45.0, 130.0)
        elif is_idh_session and t == idh_step - 1:
            if is_subtle_positive:
                # Subtle prodrome: milder SBP drop (~6-10 mmHg)
                session_sbp[t] = session_sbp[t - 1] - float(rng.uniform(6.0, 10.0)) + noise_s
                session_dbp[t] = session_dbp[t - 1] - float(rng.uniform(2.0, 5.0)) + noise_d
                session_hr[t] = session_hr[t - 1] + float(rng.uniform(2.0, 6.0)) + noise_h
            else:
                # Classic prodrome: noticeable early drop in SBP (~10-16 mmHg)
                session_sbp[t] = session_sbp[t - 1] - float(rng.uniform(10.0, 16.0)) + noise_s
                session_dbp[t] = session_dbp[t - 1] - float(rng.uniform(3.0, 8.0)) + noise_d
                session_hr[t] = session_hr[t - 1] + float(rng.uniform(3.0, 9.0)) + noise_h
        elif is_borderline_negative and t == dip_step:
            # Transient physiological dip in negative sessions: SBP drops 11-16 mmHg, but stabilizes above 96 mmHg
            session_sbp[t] = max(96.0, session_sbp[t - 1] - float(rng.uniform(11.0, 16.0)) + noise_s)
            session_dbp[t] = max(58.0, session_dbp[t - 1] - float(rng.uniform(4.0, 8.0)) + noise_d)
            session_hr[t] = session_hr[t - 1] + float(rng.uniform(4.0, 10.0)) + noise_h
        elif is_borderline_negative and t == dip_step + 1:
            # Compensatory recovery from transient dip
            session_sbp[t] = session_sbp[t - 1] + float(rng.uniform(3.0, 7.0)) + noise_s
            session_dbp[t] = session_dbp[t - 1] + float(rng.uniform(1.0, 4.0)) + noise_d
            session_hr[t] = session_hr[t - 1] - float(rng.uniform(2.0, 5.0)) + noise_h
        else:
            # Stable hemodialysis trajectory with mild drift and natural fluctuation
            sbp_drift = float(rng.normal(-0.9, 2.2))
            session_sbp[t] = max(95.0, session_sbp[t - 1] + sbp_drift)
            dbp_drift = float(rng.normal(-0.5, 1.6))
            session_dbp[t] = max(55.0, session_dbp[t - 1] + dbp_drift)
            hr_drift = float(rng.normal(0.2, 1.8))
            session_hr[t] = np.clip(session_hr[t - 1] + hr_drift, 55.0, 120.0)

        # Derived dynamic features
        session_map[t] = session_dbp[t] + (session_sbp[t] - session_dbp[t]) / 3.0
        session_sbp_drop[t] = base_sbp - session_sbp[t]

    # Construct full 8-dimensional session matrix of shape (9, 8)
    session_matrix = np.column_stack([
        session_sbp,
        session_dbp,
        session_map,
        session_sbp_drop,
        session_hr,
        session_ufr,
        session_cum_uf_pct,
        time_steps,
    ]).astype(np.float32)

    # Slice into sliding windows of length seq_len (3)
    # Window [i : i + seq_len] predicts target at step i + seq_len (t+1)
    windows = []
    labels = []
    for i in range(n_steps - seq_len):
        target_step = i + seq_len
        target_sbp = session_sbp[target_step]
        target_map = session_map[target_step]

        # Standard clinical IDH definition (KDOQI / European Best Practice Guidelines):
        # SBP < 90 mmHg OR (base_sbp - target_sbp >= 20 mmHg AND target_map < 72 mmHg)
        is_idh_event = int(target_sbp < 90.0 or (base_sbp - target_sbp >= 20.0 and target_map < 72.0))

        windows.append(session_matrix[i : i + seq_len])
        labels.append(is_idh_event)

    return windows, labels


def generate_hemodialysis_sessions(
    n_sessions: int = 1200,
    idh_ratio: float = 0.28,
    seed: int = 42,
    seq_len: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate all windows across simulated sessions (convenience helper).
    Returns:
        all_sequences: np.ndarray of shape (total_samples, seq_len, 8)
        all_labels: np.ndarray of shape (total_samples,)
    """
    rng = np.random.RandomState(seed)
    all_windows = []
    all_labels = []

    for s_id in range(n_sessions):
        is_idh = bool(rng.rand() < idh_ratio)
        windows, labels = _simulate_single_session(s_id, is_idh, rng, seq_len=seq_len)
        all_windows.extend(windows)
        all_labels.extend(labels)

    return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.float32)


def prepare_dataset_splits(
    n_sessions: int = 1200,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    seq_len: int = 3,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Zero-Data-Leakage Split Strategy:
    1. Splits 1,200 hemodialysis sessions by session_id into:
       - Train: 840 sessions (70%)
       - Validation: 180 sessions (15%)
       - Test: 180 sessions (15%)
    2. Slices sliding windows (seq_len = 3) strictly within each partition.
    3. Fits StandardScaler ONLY on X_train, then applies to X_val and X_test.
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_dict, split_metadata
    """
    rng = np.random.RandomState(seed)

    # Assign sessions into 3 independent sets
    session_ids = np.arange(n_sessions)
    rng.shuffle(session_ids)

    n_train = int(n_sessions * train_ratio)
    n_val = int(n_sessions * val_ratio)
    n_test = n_sessions - n_train - n_val

    train_ids = set(session_ids[:n_train])
    val_ids = set(session_ids[n_train : n_train + n_val])
    test_ids = set(session_ids[n_train + n_val :])

    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []

    # Deterministic simulation with consistent random state
    sim_rng = np.random.RandomState(seed + 100)
    for s_id in range(n_sessions):
        is_idh = bool(sim_rng.rand() < 0.28)
        windows, labels = _simulate_single_session(s_id, is_idh, sim_rng, seq_len=seq_len)

        if s_id in train_ids:
            train_windows.extend(windows)
            train_labels.extend(labels)
        elif s_id in val_ids:
            val_windows.extend(windows)
            val_labels.extend(labels)
        else:
            test_windows.extend(windows)
            test_labels.extend(labels)

    X_train_raw = np.array(train_windows, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.float32)

    X_val_raw = np.array(val_windows, dtype=np.float32)
    y_val = np.array(val_labels, dtype=np.float32)

    X_test_raw = np.array(test_windows, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.float32)

    n_features = len(FEATURE_NAMES)

    # Fit scaler strictly on Train data only (Prevents data leakage)
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

    split_metadata = {
        'split_strategy': 'Session-level split (Zero time-series window leakage)',
        'train_sessions': n_train,
        'val_sessions': n_val,
        'test_sessions': n_test,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'positive_rate_train': round(float(np.mean(y_train)), 4),
        'positive_rate_val': round(float(np.mean(y_val)), 4),
        'positive_rate_test': round(float(np.mean(y_test)), 4),
    }

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_dict, split_metadata
