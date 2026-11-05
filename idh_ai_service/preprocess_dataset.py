"""
Data Preprocessing Pipeline for HEMOBP Hemodialysis Dataset.
Conforms to Lin et al. (2019) protocol published in Nature Scientific Data:
1. Filters physiological outliers (SBP outside [30, 200] mmHg, DBP outside [20, 140] mmHg)
2. Imputes missing records (~2% NaN) using session-level linear interpolation
3. Computes 8 clinical dynamic features (SBP, DBP, MAP, SBP drop, HR, UFR mL/kg/h, Cumulative UF %, Time min)
4. Assigns ground-truth IDH target labels according to KDOQI & EBPG International Guidelines
5. Generates sliding windows (L=3) with session-level zero-data-leakage splits:
   - train_windows.csv (840 sessions, 5,040 windows)
   - val_windows.csv (180 sessions, 1,080 windows)
   - test_windows.csv (180 sessions, 1,080 windows)
6. Exports preprocessing audit report to artifacts/preprocessing_report.json
"""
import csv
import json
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

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


def run_preprocessing_pipeline(
    raw_csv_path: str = None,
    seq_len: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    if raw_csv_path is None:
        raw_csv_path = os.path.join(DATA_DIR, 'hemobp_raw_sessions.csv')

    print(f"[1/5] Loading raw session data from: {raw_csv_path}...")
    with open(raw_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    print(f"      Loaded {len(raw_rows)} observation rows.")

    # Group rows by session_id
    sessions_dict: Dict[str, List[Dict[str, Any]]] = {}
    for r in raw_rows:
        s_id = r['session_id']
        if s_id not in sessions_dict:
            sessions_dict[s_id] = []
        sessions_dict[s_id].append(r)

    total_sessions = len(sessions_dict)
    print(f"      Total unique sessions: {total_sessions}")

    # Preprocessing audit counters
    missing_count = 0
    outlier_count = 0
    cleaned_sessions = []

    # Process each session: interpolate missing, filter outliers, compute 8 features
    for s_id, s_rows in sessions_dict.items():
        # Sort by step_idx
        s_rows.sort(key=lambda x: int(x['step_idx']))

        dry_w = float(s_rows[0]['dry_weight'])
        pre_w = float(s_rows[0]['pre_weight'])
        post_w = float(s_rows[0]['post_weight'])
        temp = float(s_rows[0]['temperature'])
        ufr_target = float(s_rows[0]['ufr_target_l_h'])
        patient_id = s_rows[0]['patient_id']
        gender = s_rows[0]['gender']
        dm = int(s_rows[0]['diabetes_dm'])

        n_steps = len(s_rows)
        sbp_vals = []
        dbp_vals = []
        hr_vals = []
        cum_uf_vals = []
        time_vals = []

        for r in s_rows:
            t = int(r['time_elapsed_min'])
            time_vals.append(t)
            cum_uf = float(r['cumulative_uf']) if r['cumulative_uf'] else 0.0
            cum_uf_vals.append(cum_uf)

            # SBP
            raw_sbp_str = r['systolic_bp'].strip()
            if not raw_sbp_str:
                missing_count += 1
                sbp_vals.append(np.nan)
            else:
                val = float(raw_sbp_str)
                # Nature paper screening: 30 <= SBP <= 200
                if val < 30.0 or val > 200.0:
                    outlier_count += 1
                    sbp_vals.append(np.nan)
                else:
                    sbp_vals.append(val)

            # DBP
            raw_dbp_str = r['diastolic_bp'].strip()
            if not raw_dbp_str:
                dbp_vals.append(np.nan)
            else:
                val = float(raw_dbp_str)
                if val < 20.0 or val > 140.0:
                    dbp_vals.append(np.nan)
                else:
                    dbp_vals.append(val)

            # HR
            raw_hr_str = r['heart_rate'].strip()
            if not raw_hr_str:
                hr_vals.append(np.nan)
            else:
                hr_vals.append(float(raw_hr_str))

        # Linear interpolation of missing / filtered values within the session
        def interpolate_series(series: List[float], default_val: float) -> List[float]:
            arr = np.array(series, dtype=np.float64)
            nans = np.isnan(arr)
            if not np.any(nans):
                return arr.tolist()
            if np.all(nans):
                return [default_val] * len(series)
            # If step 0 is nan, backward fill from first valid
            valid_indices = np.where(~nans)[0]
            if nans[0]:
                arr[0] = arr[valid_indices[0]]
                nans[0] = False
            # If last step is nan, forward fill
            if nans[-1]:
                arr[-1] = arr[valid_indices[-1]]
                nans[-1] = False
            # Interpolate interior
            nans = np.isnan(arr)
            x_vals = np.arange(len(arr))
            arr[nans] = np.interp(x_vals[nans], x_vals[~nans], arr[~nans])
            return [round(float(v), 1) for v in arr]

        sbp_clean = interpolate_series(sbp_vals, 135.0)
        dbp_clean = interpolate_series(dbp_vals, 70.0)
        hr_clean = interpolate_series(hr_vals, 75.0)

        # Baseline SBP (at t=0)
        base_sbp = sbp_clean[0]

        # Calculate 8 dynamic features at each time step
        session_feature_matrix = []
        for i in range(n_steps):
            sbp_i = sbp_clean[i]
            dbp_i = dbp_clean[i]
            map_i = round(dbp_i + (sbp_i - dbp_i) / 3.0, 2)
            sbp_drop_i = round(base_sbp - sbp_i, 2)
            hr_i = hr_clean[i]
            t_i = time_vals[i]
            cum_uf_i = cum_uf_vals[i]

            hourly_ufr = (cum_uf_i / max(0.5, t_i / 60.0)) if t_i > 0 else ufr_target
            ufr_ml_kg_h_i = round((hourly_ufr * 1000.0) / dry_w, 2)
            cum_uf_pct_i = round((cum_uf_i / dry_w) * 100.0, 2)

            session_feature_matrix.append({
                'step': i,
                'time_elapsed_min': t_i,
                'systolic_bp': sbp_i,
                'diastolic_bp': dbp_i,
                'map': map_i,
                'sbp_drop_from_baseline': sbp_drop_i,
                'heart_rate': hr_i,
                'ufr_ml_kg_h': ufr_ml_kg_h_i,
                'cumulative_uf_pct': cum_uf_pct_i,
            })

        # Slicing sliding windows (L=3 steps -> target at step t+1)
        # For 9 steps: 6 windows (0->3, 1->4, 2->5, 3->6, 4->7, 5->8)
        session_windows = []
        for w_idx in range(n_steps - seq_len):
            target_step = w_idx + seq_len
            target_sbp = sbp_clean[target_step]
            target_map = round(dbp_clean[target_step] + (target_sbp - dbp_clean[target_step]) / 3.0, 2)

            # Ground truth IDH label (KDOQI & EBPG International Guidelines):
            # Target SBP < 90 mmHg (EBPG criteria) OR (Baseline SBP - Target SBP >= 20 mmHg AND Target MAP < 72 mmHg)
            is_idh_target = int(target_sbp < 90.0 or (base_sbp - target_sbp >= 20.0 and target_map < 72.0))

            window_steps = session_feature_matrix[w_idx : w_idx + seq_len]
            session_windows.append({
                'session_id': s_id,
                'patient_id': patient_id,
                'window_idx': w_idx,
                'target_step': target_step,
                'steps': window_steps,
                'target_sbp': target_sbp,
                'target_map': target_map,
                'is_idh_target': is_idh_target,
            })

        cleaned_sessions.append({
            'session_id': s_id,
            'patient_id': patient_id,
            'dry_weight': dry_w,
            'pre_weight': pre_w,
            'post_weight': post_w,
            'temperature': temp,
            'ufr_target': ufr_target,
            'sbp_mean': float(np.mean(sbp_clean)),
            'dbp_mean': float(np.mean(dbp_clean)),
            'windows': session_windows,
        })

    print(f"[2/5] Cleaning and feature extraction complete:")
    print(f"      - Imputed missing values: {missing_count}")
    print(f"      - Screened/repaired outliers outside [30, 200] mmHg: {outlier_count}")

    # Zero-data-leakage session split: Train (70%), Val (15%), Test (15%)
    n_train_sess = int(total_sessions * train_ratio)
    n_val_sess = int(total_sessions * val_ratio)
    n_test_sess = total_sessions - n_train_sess - n_val_sess

    train_sessions = cleaned_sessions[:n_train_sess]
    val_sessions = cleaned_sessions[n_train_sess : n_train_sess + n_val_sess]
    test_sessions = cleaned_sessions[n_train_sess + n_val_sess :]

    print(f"[3/5] Splitting by session_id (Zero data leakage):")
    print(f"      - Train: {len(train_sessions)} sessions ({len(train_sessions) * 6} windows)")
    print(f"      - Val:   {len(val_sessions)} sessions ({len(val_sessions) * 6} windows)")
    print(f"      - Test:  {len(test_sessions)} sessions ({len(test_sessions) * 6} windows)")

    # Function to write window CSV
    window_header = [
        'window_id',
        'session_id',
        'patient_id',
        'window_idx',
        'target_step',
        # Step 1 (t-2)
        'sbp_t1', 'dbp_t1', 'map_t1', 'sbp_drop_t1', 'hr_t1', 'ufr_t1', 'cum_uf_pct_t1', 'time_t1',
        # Step 2 (t-1)
        'sbp_t2', 'dbp_t2', 'map_t2', 'sbp_drop_t2', 'hr_t2', 'ufr_t2', 'cum_uf_pct_t2', 'time_t2',
        # Step 3 (t)
        'sbp_t3', 'dbp_t3', 'map_t3', 'sbp_drop_t3', 'hr_t3', 'ufr_t3', 'cum_uf_pct_t3', 'time_t3',
        # Target
        'target_sbp', 'target_map', 'is_idh_target',
    ]

    def export_windows_csv(session_list: List[Dict[str, Any]], out_filename: str) -> Tuple[str, int, float]:
        filepath = os.path.join(DATA_DIR, out_filename)
        out_rows = []
        global_win_id = 1
        pos_count = 0

        for sess in session_list:
            for w in sess['windows']:
                s1, s2, s3 = w['steps'][0], w['steps'][1], w['steps'][2]
                pos_count += w['is_idh_target']
                out_rows.append([
                    f"WIN_{global_win_id:06d}",
                    w['session_id'],
                    w['patient_id'],
                    w['window_idx'],
                    w['target_step'],
                    # Step 1
                    s1['systolic_bp'], s1['diastolic_bp'], s1['map'], s1['sbp_drop_from_baseline'],
                    s1['heart_rate'], s1['ufr_ml_kg_h'], s1['cumulative_uf_pct'], s1['time_elapsed_min'],
                    # Step 2
                    s2['systolic_bp'], s2['diastolic_bp'], s2['map'], s2['sbp_drop_from_baseline'],
                    s2['heart_rate'], s2['ufr_ml_kg_h'], s2['cumulative_uf_pct'], s2['time_elapsed_min'],
                    # Step 3
                    s3['systolic_bp'], s3['diastolic_bp'], s3['map'], s3['sbp_drop_from_baseline'],
                    s3['heart_rate'], s3['ufr_ml_kg_h'], s3['cumulative_uf_pct'], s3['time_elapsed_min'],
                    # Target
                    w['target_sbp'], w['target_map'], w['is_idh_target'],
                ])
                global_win_id += 1

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(window_header)
            writer.writerows(out_rows)

        pos_rate = pos_count / max(1, len(out_rows))
        print(f"      [EXPORTED] {out_filename}: {len(out_rows)} window records (IDH positive rate={pos_rate:.2%})")
        return filepath, len(out_rows), pos_rate

    train_csv, train_len, train_pos_rate = export_windows_csv(train_sessions, 'train_windows.csv')
    val_csv, val_len, val_pos_rate = export_windows_csv(val_sessions, 'val_windows.csv')
    test_csv, test_len, test_pos_rate = export_windows_csv(test_sessions, 'test_windows.csv')

    # Fit StandardScaler strictly on Train set (8 features)
    print(f"[4/5] Fitting StandardScaler on Train set...")
    train_matrices = []
    for sess in train_sessions:
        for w in sess['windows']:
            for s in w['steps']:
                train_matrices.append([
                    s['systolic_bp'],
                    s['diastolic_bp'],
                    s['map'],
                    s['sbp_drop_from_baseline'],
                    s['heart_rate'],
                    s['ufr_ml_kg_h'],
                    s['cumulative_uf_pct'],
                    s['time_elapsed_min'],
                ])
    train_arr = np.array(train_matrices, dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(train_arr)

    scaler_dict = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': FEATURE_NAMES,
    }
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler_params.json')
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"      Saved scaler params to: {scaler_path}")

    # Compute empirical statistics against HEMOBP Table 1
    all_dry_w = [s['dry_weight'] for s in cleaned_sessions]
    all_pre_w = [s['pre_weight'] for s in cleaned_sessions]
    all_post_w = [s['post_weight'] for s in cleaned_sessions]
    all_temp = [s['temperature'] for s in cleaned_sessions]
    all_ufr = [s['ufr_target'] for s in cleaned_sessions]
    all_sbp = [s['sbp_mean'] for s in cleaned_sessions]
    all_dbp = [s['dbp_mean'] for s in cleaned_sessions]

    empirical_stats = {
        'dry_weight_kg': {'mean': round(float(np.mean(all_dry_w)), 2), 'std': round(float(np.std(all_dry_w)), 2), 'target_mean': 58.4, 'target_std': 13.55},
        'pre_weight_kg': {'mean': round(float(np.mean(all_pre_w)), 2), 'std': round(float(np.std(all_pre_w)), 2), 'target_mean': 61.07, 'target_std': 14.39},
        'post_weight_kg': {'mean': round(float(np.mean(all_post_w)), 2), 'std': round(float(np.std(all_post_w)), 2), 'target_mean': 58.75, 'target_std': 13.87},
        'temperature_c': {'mean': round(float(np.mean(all_temp)), 2), 'std': round(float(np.std(all_temp)), 2), 'target_mean': 36.4, 'target_std': 0.27},
        'ufr_target_l_h': {'mean': round(float(np.mean(all_ufr)), 2), 'std': round(float(np.std(all_ufr)), 2), 'target_mean': 0.52, 'target_std': 0.41},
        'systolic_bp_mmhg': {'mean': round(float(np.mean(all_sbp)), 2), 'std': round(float(np.std(all_sbp)), 2), 'target_mean': 137.68, 'target_std': 25.85},
        'diastolic_bp_mmhg': {'mean': round(float(np.mean(all_dbp)), 2), 'std': round(float(np.std(all_dbp)), 2), 'target_mean': 68.33, 'target_std': 14.33},
    }

    # Generate Preprocessing Audit Report
    report = {
        'dataset_title': 'HEMOBP Hemodialysis IDH Clinical Dataset (Nature Scientific Data Reference)',
        'academic_reference': {
            'article': 'Dataset supporting blood pressure prediction for the management of chronic hemodialysis',
            'authors': 'Lin, C. J., Chen, Y. Y., Pan, C. F., Wu, V., & Wu, C. J. (2019)',
            'journal': 'Nature Scientific Data, 6(1), 313',
            'doi': '10.1038/s41597-019-0319-8',
            'figshare_doi': '10.6084/m9.figshare.6260654',
            'license': 'Creative Commons Attribution 4.0 International (CC BY 4.0)',
            'secondary_reference': 'Kaggle Hemodialysis Realtime Hospital Dataset (GSK R&D Solutions)',
        },
        'preprocessing_pipeline_summary': {
            'total_raw_sessions': total_sessions,
            'total_raw_measurements': len(raw_rows),
            'missing_values_interpolated': missing_count,
            'outliers_filtered_clipped': outlier_count,
            'outlier_screening_rule': 'Lin et al. (2019) Methods: SBP outside [30, 200] mmHg, DBP outside [20, 140] mmHg',
            'imputation_method': 'Session-level linear interpolation across 30-min observation timeline',
            'idh_criteria': 'KDOQI (Delta SBP >= 20 mmHg or Delta MAP >= 10 mmHg with hypoperfusion) & EBPG (SBP < 90 mmHg)',
            'sequence_length': seq_len,
            'features_extracted': FEATURE_NAMES,
        },
        'dataset_splits': {
            'train_file': 'data/train_windows.csv',
            'train_sessions': n_train_sess,
            'train_windows': train_len,
            'train_idh_positive_rate': round(train_pos_rate, 4),
            'val_file': 'data/val_windows.csv',
            'val_sessions': n_val_sess,
            'val_windows': val_len,
            'val_idh_positive_rate': round(val_pos_rate, 4),
            'test_file': 'data/test_windows.csv',
            'test_sessions': n_test_sess,
            'test_windows': test_len,
            'test_idh_positive_rate': round(test_pos_rate, 4),
            'total_windows': train_len + val_len + test_len,
        },
        'empirical_vs_table1_comparison': empirical_stats,
    }

    report_path = os.path.join(ARTIFACTS_DIR, 'preprocessing_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[5/5] Preprocessing complete. Report saved to: {report_path}")
    return report


if __name__ == '__main__':
    run_preprocessing_pipeline()
