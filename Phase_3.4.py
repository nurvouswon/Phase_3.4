import streamlit as st
import pandas as pd
import numpy as np
import gc
import time
import psutil
from datetime import timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration

import matplotlib.pyplot as plt
import shap

from scipy.special import logit, expit

# ===================== UI =====================
st.set_page_config(page_title="🏆 MLB Home Run Predictor — State of the Art, Phase 1", layout="wide")
st.title("🏆 MLB Home Run Predictor — State of the Art, Phase 1")

# ===================== Helpers / Utilities =====================
@st.cache_data(show_spinner=False, max_entries=2)
def safe_read_cached(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith('.parquet'):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', low_memory=False)

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def find_duplicate_columns(df):
    return [col for col in df.columns if list(df.columns).count(col) > 1]

def fix_types(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        if df[col].dtype == 'O':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype(pd.Int64Dtype())
    return df

def clean_X(df, train_cols=None):
    df = dedup_columns(df)
    df = fix_types(df)
    allowed_obj = {'wind_dir_string', 'condition', 'player_name', 'city', 'park', 'roof_status', 'team_code', 'time'}
    drop_cols = [c for c in df.select_dtypes('O').columns if c not in allowed_obj]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.fillna(-1)
    if train_cols is not None:
        for c in train_cols:
            if c not in df.columns:
                df[c] = -1
        df = df[list(train_cols)]
    return df

def get_valid_feature_cols(df, drop=None):
    base_drop = set(['game_date','batter_id','player_name','pitcher_id','city','park','roof_status','team_code','time'])
    if drop: base_drop = base_drop.union(drop)
    numerics = df.select_dtypes(include=[np.number]).columns
    return [c for c in numerics if c not in base_drop]

def nan_inf_check(X, name):
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])
        nans = X_num.isna().sum().sum()
        infs = np.isinf(X_num.to_numpy(dtype=np.float64, copy=False)).sum()
    else:
        nans = np.isnan(X).sum()
        infs = np.isinf(X).sum()
    if nans > 0 or infs > 0:
        st.error(f"Found {nans} NaNs and {infs} Infs in {name}! Please fix.")
        st.stop()

def feature_debug(X):
    X.columns = X.columns.astype(str)  # Ensure column names are strings
    st.write("🛡️ Feature Debugging:")
    st.write("Data types:", X.dtypes.value_counts())
    object_cols = X.select_dtypes(include="object").columns.tolist()
    if object_cols:
        st.write("Columns with object dtype:", object_cols)
    else:
        st.write("Columns with object dtype: []")
    for col in X.columns:
        try:
            if X[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                st.write(f"Column `{col}` is {X[col].dtype}, unique values: {X[col].unique()[:8]}")
        except Exception as e:
            st.write(f"⚠️ Could not inspect column `{col}`: {e}")
    st.write("Missing values per column (top 10):", X.isna().sum().sort_values(ascending=False).head(10))

def drift_check(train, today, n=5):
    drifted = []
    for c in train.columns:
        if c not in today.columns: continue
        tmean = np.nanmean(train[c])
        tstd = np.nanstd(train[c])
        dmean = np.nanmean(today[c])
        if tstd > 0 and abs(tmean - dmean) / tstd > n:
            drifted.append(c)
    return drifted

def winsorize_clip(X, limits=(0.01, 0.99)):
    X = X.astype(float)
    for col in X.columns:
        lower = X[col].quantile(limits[0])
        upper = X[col].quantile(limits[1])
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X

def remove_outliers(
    X,
    y,
    method="iforest",
    contamination=0.012,
    n_estimators=100,
    max_samples='auto',
    n_neighbors=20,
    scale=True
):
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    if method == "iforest":
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        mask = clf.fit_predict(X_scaled) == 1
    elif method == "lof":
        clf = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors
        )
        mask = clf.fit_predict(X_scaled) == 1
    else:
        raise ValueError("Unknown method: choose 'iforest' or 'lof'")
    return X[mask], y[mask]

def smooth_labels(y, smoothing=0.02):
    y = np.asarray(y)
    y_smooth = y.copy().astype(float)
    y_smooth[y == 1] = 1 - smoothing
    y_smooth[y == 0] = smoothing
    return y_smooth

# ===================== Overlay & Weather (UPGRADED) =====================
def _getv(row, keys, default=np.nan):
    for k in keys:
        if k in row and pd.notnull(row[k]):
            return row[k]
    return default

def _hand(row, batter=True):
    return str(_getv(row, ["stand","batter_hand"] if batter else ["pitcher_hand","p_throws"], "R")).upper() or "R"

def _first_present(row, base, windows):
    for w in windows:
        col = f"{base}_{w}"
        if col in row and pd.notnull(row[col]):
            return row[col]
    return np.nan

def overlay_multiplier(row):
    EDGE_MIN, EDGE_MAX = 0.68, 1.44
    PULL_HI, PULL_LO   = 0.35, 0.28
    FB_HI_P, FB_HI_B   = 0.25, 0.22
    BARREL_HI, BARREL_MID = 0.12, 0.08
    HOT_HR_HI, HOT_HR_LO = 0.09, 0.025

    edge = 1.0

    b_hand = _hand(row, True)
    p_hand = _hand(row, False)

    temp     = _getv(row, ["temp"])
    humidity = _getv(row, ["humidity"])
    wind     = _getv(row, ["wind_mph"])
    wind_dir = str(_getv(row, ["wind_dir_string"], "")).lower().strip()
    roof     = str(_getv(row, ["roof_status"], "")).lower().strip()
    altitude = _getv(row, ["park_altitude"])

    pf_base  = _getv(row, ["park_hr_rate", "park_hand_hr_rate", "park_hr_pct_hand"])
    pf_rhb   = _getv(row, ["park_hr_pct_rhb"])
    pf_lhb   = _getv(row, ["park_hr_pct_lhb"])
    pf_hand  = pf_rhb if b_hand == "R" else pf_lhb if b_hand == "L" else np.nan

    def _cap_pf(x):
        try: return max(0.80, min(1.22, float(x)))
        except Exception: return np.nan

    pfs = [pf_hand, pf_base]
    pfs = [_cap_pf(x) for x in pfs if pd.notnull(x)]
    if pfs:
        edge *= pfs[0]

    b_pull  = _first_present(row, "b_pull_rate",   [7,14,5,3])
    b_fb    = _first_present(row, "b_fb_rate",     [7,14,5,3])
    b_brl   = _first_present(row, "b_barrel_rate", [7,14,5,3])
    b_hot   = _first_present(row, "b_hr_per_pa",   [7,5,3])

    if pd.notnull(b_brl):
        if b_brl >= BARREL_HI: edge *= 1.04
        elif b_brl >= BARREL_MID: edge *= 1.02

    if pd.notnull(b_hot):
        if b_hot > HOT_HR_HI: edge *= 1.04
        elif b_hot < HOT_HR_LO: edge *= 0.97

    p_fb = _first_present(row, "p_fb_rate", [14,7,5])

    roof_closed = ("closed" in roof) or ("indoor" in roof) or ("domed" in roof)
    if pd.notnull(altitude):
        if altitude >= 5000: edge *= 1.05
        elif altitude >= 3000: edge *= 1.02

    if pd.notnull(temp):
        edge *= 1.035 ** ((temp - 70) / 10.0)
    if pd.notnull(humidity):
        if humidity >= 65: edge *= 1.02
        elif humidity <= 35: edge *= 0.98

    pulled_field = "lf" if b_hand == "R" else "rf"
    oppo_field   = "rf" if b_hand == "R" else "lf"

    wind_factor = 1.0
    if pd.notnull(wind) and wind >= 6 and wind_dir:
        out = ("out" in wind_dir)
        inn = ("in"  in wind_dir)
        has_lf = "lf" in wind_dir
        has_rf = "rf" in wind_dir
        has_cf = ("cf" in wind_dir) or ("center" in wind_dir)

        hi_pull = pd.notnull(b_pull) and (b_pull >= PULL_HI)
        lo_pull = pd.notnull(b_pull) and (b_pull <= PULL_LO)
        hi_bfb  = pd.notnull(b_fb)   and (b_fb   >= FB_HI_B)
        hi_pfb  = pd.notnull(p_fb)   and (p_fb   >= FB_HI_P)

        OUT_CF_BOOST   = 1.11
        OUT_PULL_BOOST = 1.20
        OPPO_TINY      = 1.05
        IN_CF_FADE     = 0.92
        IN_PULL_FADE   = 0.85

        if has_cf and hi_bfb:
            wind_factor *= OUT_CF_BOOST if out else IN_CF_FADE if inn else 1.0

        if has_lf and pulled_field == "lf" and hi_pull:
            wind_factor *= OUT_PULL_BOOST if out else IN_PULL_FADE if inn else 1.0
        if has_rf and pulled_field == "rf" and hi_pull:
            wind_factor *= OUT_PULL_BOOST if out else IN_PULL_FADE if inn else 1.0

        if out and lo_pull:
            if has_lf and oppo_field == "lf": wind_factor *= OPPO_TINY
            if has_rf and oppo_field == "rf": wind_factor *= OPPO_TINY

        if hi_pfb and (out or inn):
            wind_factor *= 1.05 if out else 0.97

        if roof_closed:
            wind_factor = 1.0 + (wind_factor - 1.0) * 0.35

        if out or inn:
            extra = max(0.0, (wind - 8.0) / 3.0)
            wind_factor *= min(1.08, 1.0 + 0.01 * extra) if out else max(0.92, 1.0 - 0.01 * extra)

    edge *= wind_factor

    if pd.notnull(temp) and pd.notnull(wind):
        if (temp >= 75 and wind >= 7 and "out" in wind_dir) and not roof_closed:
            edge *= 1.05
        elif temp >= 65 and wind >= 5 and not roof_closed:
            edge *= 1.02
        else:
            edge *= 0.985

    if b_hand != p_hand: edge *= 1.01
    else: edge *= 0.995

    return float(np.clip(edge, EDGE_MIN, EDGE_MAX))

def rate_weather(row):
    ratings = {}
    temp = _getv(row, ["temp"])
    if pd.isna(temp):
        ratings["temp_rating"] = "?"
    elif 75 <= temp <= 85:
        ratings["temp_rating"] = "Excellent"
    elif 68 <= temp < 75 or 85 < temp <= 90:
        ratings["temp_rating"] = "Good"
    elif 60 <= temp < 68 or 90 < temp <= 95:
        ratings["temp_rating"] = "Fair"
    else:
        ratings["temp_rating"] = "Poor"

    hum = _getv(row, ["humidity"])
    if pd.isna(hum):
        ratings["humidity_rating"] = "?"
    elif hum >= 60:
        ratings["humidity_rating"] = "Excellent"
    elif 45 <= hum < 60:
        ratings["humidity_rating"] = "Good"
    elif 30 <= hum < 45:
        ratings["humidity_rating"] = "Fair"
    else:
        ratings["humidity_rating"] = "Poor"

    wind = _getv(row, ["wind_mph"])
    wdir = str(_getv(row, ["wind_dir_string"], "")).lower()
    if pd.isna(wind):
        ratings["wind_rating"] = "?"
    elif wind < 6:
        ratings["wind_rating"] = "Excellent"
    elif 6 <= wind < 12:
        ratings["wind_rating"] = "Good"
    elif 12 <= wind < 18:
        if "out" in wdir:
            ratings["wind_rating"] = "Good"
        elif "in" in wdir:
            ratings["wind_rating"] = "Fair"
        else:
            ratings["wind_rating"] = "Fair"
    else:
        if "out" in wdir:
            ratings["wind_rating"] = "Fair"
        elif "in" in wdir:
            ratings["wind_rating"] = "Poor"
        else:
            ratings["wind_rating"] = "Poor"

    cond = str(_getv(row, ["condition"], "")).lower()
    if not cond or cond in ("unknown","none","na"):
        ratings["condition_rating"] = "?"
    elif "clear" in cond or "sun" in cond or "outdoor" in cond:
        ratings["condition_rating"] = "Excellent"
    elif "cloud" in cond or "partly" in cond:
        ratings["condition_rating"] = "Good"
    elif "rain" in cond or "fog" in cond:
        ratings["condition_rating"] = "Poor"
    else:
        ratings["condition_rating"] = "Fair"
    return pd.Series(ratings)

# ===================== APP START =====================
event_file = st.file_uploader("Upload Event-Level CSV/Parquet for Training (required)", type=['csv', 'parquet'], key='eventcsv')
today_file = st.file_uploader("Upload TODAY CSV for Prediction (required)", type=['csv', 'parquet'], key='todaycsv')

if event_file is not None and today_file is not None:
    with st.spinner("Loading and prepping files..."):
        event_df = safe_read_cached(event_file)
        today_df = safe_read_cached(today_file)

        # Basic cleaning
        event_df = event_df.dropna(axis=1, how='all')
        today_df = today_df.dropna(axis=1, how='all')
        event_df = dedup_columns(event_df)
        today_df = dedup_columns(today_df)
        event_df = event_df.reset_index(drop=True)
        today_df = today_df.reset_index(drop=True)

        if find_duplicate_columns(event_df):
            st.error(f"Duplicate columns in event file")
            st.stop()
        if find_duplicate_columns(today_df):
            st.error(f"Duplicate columns in today file")
            st.stop()

        # Type fixes
        event_df = fix_types(event_df)
        today_df = fix_types(today_df)

        st.write(f"event_df shape: {event_df.shape}, today_df shape: {today_df.shape}")
        st.write(f"event_df memory usage (MB): {event_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        st.write(f"today_df memory usage (MB): {today_df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    target_col = 'hr_outcome'
    if target_col not in event_df.columns:
        st.error("ERROR: No valid hr_outcome column found in event-level file.")
        st.stop()
    st.success("✅ 'hr_outcome' column found in event-level data.")

    # ---- Feature Filtering ----
    # Drop known leakage / post hoc columns explicitly
    LEAK = {
        "post_away_score","post_home_score","post_bat_score","post_fld_score",
        "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
        "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
        "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
        "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
        "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
    }
    event_df = event_df.drop(columns=[c for c in event_df.columns if c in LEAK], errors="ignore")

    feature_cols = sorted(list(set(get_valid_feature_cols(event_df)) & set(get_valid_feature_cols(today_df))))
    st.write(f"Feature count before filtering: {len(feature_cols)}")

    X = clean_X(event_df[feature_cols])
    X_today = clean_X(today_df[feature_cols], train_cols=X.columns)

    feature_debug(X)

    nan_thresh = 0.3
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > nan_thresh].index.tolist()
    if drop_cols:
        st.warning(f"Dropping {len(drop_cols)} features with >30% NaNs: {drop_cols[:20]}")
        X = X.drop(columns=drop_cols)
        X_today = X_today.drop(columns=drop_cols, errors='ignore')

    nzv_cols = X.loc[:, X.nunique() <= 2].columns.tolist()
    if nzv_cols:
        st.warning(f"Dropping {len(nzv_cols)} near-constant features.")
        X = X.drop(columns=nzv_cols)
        X_today = X_today.drop(columns=nzv_cols, errors='ignore')

    corrs = X.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    if to_drop:
        st.warning(f"Dropping {len(to_drop)} highly correlated features.")
        X = X.drop(columns=to_drop)
        X_today = X_today.drop(columns=to_drop, errors='ignore')

    X = winsorize_clip(X)
    X_today = winsorize_clip(X_today)

    # --- Optional: polynomial crosses (OFF by default to save memory)
    use_crosses = st.checkbox("Enable polynomial interaction crosses (slow, memory heavy)", value=False)
    if use_crosses:
        try:
            gc.collect()
            mem_before = psutil.virtual_memory().available / (1024 ** 3)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            Xt = X.astype(np.float32)
            X_cross = pd.DataFrame(poly.fit_transform(Xt), columns=poly.get_feature_names_out(X.columns))
            X_cross = dedup_columns(X_cross)
            # Keep only top-var crosses to cap memory
            var_scores = X_cross.var().sort_values(ascending=False)
            keep_cross = list(var_scores.head(min(300, max(50, X_cross.shape[1]//10))).index)
            X_cross = X_cross[keep_cross]
            X = pd.concat([X, X_cross], axis=1)
            # align today
            Xt_today = X_today.astype(np.float32)
            X_today_cross = pd.DataFrame(poly.transform(Xt_today), columns=poly.get_feature_names_out(X_today.columns))
            X_today_cross = X_today_cross[keep_cross]
            X_today = pd.concat([X_today, X_today_cross], axis=1)
            gc.collect()
            mem_after = psutil.virtual_memory().available / (1024 ** 3)
            st.write(f"🧠 Memory: {mem_before:.1f}GB → {mem_after:.1f}GB after crosses")
        except Exception as e:
            st.error(f"❌ Cross-feature generation failed: {str(e)}")

    # --- Add two cheap synergy features (tiny memory, good lift)
    def add_synergy_feats(df_ref, df_today_ref):
        for df in (df_ref, df_today_ref):
            if "b_fb_rate_7" in df.columns and "park_hr_rate" in df.columns:
                df["feat_fb_park7"] = df["b_fb_rate_7"].astype(np.float32) * df["park_hr_rate"].astype(np.float32)
            elif "b_fb_rate_14" in df.columns and "park_hr_rate" in df.columns:
                df["feat_fb_park7"] = df["b_fb_rate_14"].astype(np.float32) * df["park_hr_rate"].astype(np.float32)
            else:
                df["feat_fb_park7"] = 0.0
            temp_col = "temp" if "temp" in df.columns else None
            pfb_col = "p_fb_rate_14" if "p_fb_rate_14" in df.columns else ("p_fb_rate_7" if "p_fb_rate_7" in df.columns else None)
            if temp_col and pfb_col:
                df["feat_pfb_temp"] = df[pfb_col].astype(np.float32) * np.maximum(0, df[temp_col].astype(np.float32) - 70.0)
            else:
                df["feat_pfb_temp"] = 0.0
        return df_ref, df_today_ref

    X, X_today = add_synergy_feats(X, X_today)
    # ---- Time-aware ordering BEFORE outlier removal ----
    y = event_df[target_col].astype(int)

    if "game_date" in event_df.columns:
    # Convert to datetime safely
        dates = pd.to_datetime(event_df["game_date"], errors="coerce")
    # Fill NaT with earliest observed date to keep them at the start (stable)
        min_date = dates.min()
        if pd.isna(min_date):
        # If all NaT, just skip ordering
            st.warning("All game_date values are NaT; skipping chronological ordering.")
        else:
            dates_filled = dates.fillna(min_date)
        # Stable sort to preserve original order within same date
            order_idx = dates_filled.sort_values(kind="mergesort").index
        # Use .loc (label-based) on X and y which still align 1:1 with event_df here
            X = X.loc[order_idx].reset_index(drop=True)
            y = y.loc[order_idx].reset_index(drop=True)
    # --- Outlier removal ---
    st.write("🚦 Starting outlier removal...")
    y = event_df[target_col].astype(int)
    X, y = remove_outliers(X, y, method="iforest", contamination=0.012)
    X = X.reset_index(drop=True).copy()
    y = pd.Series(y).reset_index(drop=True)
    st.write(f"✅ Outlier removal complete. Rows remaining: {X.shape[0]}")

    # --- Fill missing values ---
    st.write("🩹 Filling missing values...")
    X = X.fillna(-1)
    X_today = X_today.fillna(-1)

    # --- Convert to float32 for memory ---
    try:
        X = X.astype(np.float32)
        X_today = X_today.astype(np.float32)
        st.success("✅ Converted feature matrices to float32")
    except Exception as e:
        st.error(f"❌ Conversion to float32 failed: {e}")

    # --- Debug view ---
    feature_debug(X_today)
    st.dataframe(X_today.head(500))  # cap preview

    st.write(f"✅ Final training matrix: {X.shape}")
    st.write("🎯 Feature engineering complete.")

    # ========== Train/Validation via Time-Aware Splits ==========
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # imbalance handling
    pos = float(y.sum())
    neg = float(len(y) - pos)
    spw = max(1.0, neg / max(1.0, pos))

    # OOF holders for stack
    P_xgb_oof = np.zeros(len(y), dtype=np.float32)
    P_lgb_oof = np.zeros(len(y), dtype=np.float32)
    P_cat_oof = np.zeros(len(y), dtype=np.float32)

    # Today preds accumulators
    P_xgb_today = []
    P_lgb_today = []
    P_cat_today = []

    fold_times = []
    show_shap = st.checkbox("Show SHAP Feature Importance (slow; small data only)", value=False)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        t_fold_start = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Smooth labels (helps with extreme rarity)
        y_tr_s = smooth_labels(y_tr.values, smoothing=0.02)

        # Base models
        xgb_clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=50,
            n_jobs=1, verbosity=0
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_lambda=2.0, n_jobs=1, is_unbalance=True
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=1500, depth=7, learning_rate=0.03, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="Logloss",
            class_weights=[1.0, spw], od_type="Iter", od_wait=50,
            verbose=0, thread_count=1
        )

        # Fit with early stopping
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.log_evaluation(0)]
        )
        cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # OOF preds
        P_xgb_oof[va_idx] = xgb_clf.predict_proba(X_va)[:,1]
        P_lgb_oof[va_idx] = lgb_clf.predict_proba(X_va)[:,1]
        P_cat_oof[va_idx] = cat_clf.predict_proba(X_va)[:,1]

        # Today preds
        P_xgb_today.append(xgb_clf.predict_proba(X_today)[:,1])
        P_lgb_today.append(lgb_clf.predict_proba(X_today)[:,1])
        P_cat_today.append(cat_clf.predict_proba(X_today)[:,1])

        # Optional SHAP (safe plotting)
        if fold == 0 and show_shap:
            with st.spinner("Computing SHAP values (this can be slow)..."):
                explainer = shap.TreeExplainer(xgb_clf)
                shap_values = explainer.shap_values(X_va)
                st.write("Top SHAP Features (XGB, validation set):")
                shap.summary_plot(shap_values, pd.DataFrame(X_va, columns=X.columns), show=False)
                fig = plt.gcf()
                st.pyplot(fig, clear_figure=True)
                plt.clf()

        fold_time = time.time() - t_fold_start
        fold_times.append(fold_time)
        avg_time = np.mean(fold_times)
        est_time_left = avg_time * (n_splits - (fold + 1))
        st.write(f"Fold {fold + 1}/{n_splits} finished in {timedelta(seconds=int(fold_time))}. Est. {timedelta(seconds=int(est_time_left))} left.")

    # Stack with meta LR
    X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
    scaler_meta = StandardScaler()
    X_meta_s = scaler_meta.fit_transform(X_meta)
    meta = LogisticRegression(max_iter=1000, solver="lbfgs")
    meta.fit(X_meta_s, y.values)

    P_today_base = np.column_stack([
        np.mean(P_xgb_today, axis=0),
        np.mean(P_lgb_today, axis=0),
        np.mean(P_cat_today, axis=0)
    ]).astype(np.float32)
    P_today_meta = meta.predict_proba(scaler_meta.transform(P_today_base))[:,1]

    # OOF performance of meta
    oof_pred_meta = meta.predict_proba(X_meta_s)[:,1]
    auc_oof = roc_auc_score(y, oof_pred_meta)
    ll_oof  = log_loss(y, oof_pred_meta)
    st.success(f"OOF Meta AUC: {auc_oof:.4f} | OOF Meta LogLoss: {ll_oof:.4f}")

    # ===== Calibration on OOF only =====
    st.markdown("### 📊 Calibration (Isotonic on OOF meta)")
    ir = IsotonicRegression(out_of_bounds="clip")
    y_oof_iso = ir.fit_transform(oof_pred_meta, y.values)
    today_iso = ir.transform(P_today_meta)

    # Optional BetaCalibration for comparison
    bc = BetaCalibration(parameters="abm")
    bc.fit(oof_pred_meta.reshape(-1,1), y.values)
    y_oof_beta = bc.predict(oof_pred_meta.reshape(-1,1))
    today_beta = bc.predict(P_today_meta.reshape(-1,1))
    # Blend (optional view)
    today_blend = 0.5 * today_iso + 0.5 * today_beta

    # Report cal metrics on OOF
    st.write(f"**Isotonic (OOF):**   AUC = {roc_auc_score(y, y_oof_iso):.4f}   |   LogLoss = {log_loss(y, y_oof_iso):.4f}")
    st.write(f"**BetaCal  (OOF):**   AUC = {roc_auc_score(y, y_oof_beta):.4f}  |   LogLoss = {log_loss(y, y_oof_beta):.4f}")

    # ---- ADD WEATHER RATINGS & OVERLAY ----
    today_df = today_df.copy()
    ratings_df = today_df.apply(rate_weather, axis=1)
    for col in ratings_df.columns:
        today_df[col] = ratings_df[col]

    # Always recompute overlay to ensure consistency
    today_df["overlay_multiplier"] = today_df.apply(overlay_multiplier, axis=1)

    # ---- Build leaderboard (rank by logit(p) + log(overlay)) ----
    def build_leaderboard(df, calibrated_probs, label="calibrated_hr_probability"):
        df = df.copy()
        df[label] = calibrated_probs
        # ranked score
        safe_p = np.clip(df[label].values, 1e-6, 1-1e-6)
        score = logit(safe_p) + np.log(df["overlay_multiplier"].clip(0.68, 1.44).values)
        df["ranked_probability"] = expit(score)

        df = df.sort_values("ranked_probability", ascending=False).reset_index(drop=True)
        df['hr_base_rank'] = df[label].rank(method='min', ascending=False)

        leaderboard_cols = []
        for c in ["player_name", "team_code", "time"]:
            if c in df.columns: leaderboard_cols.append(c)

        leaderboard_cols += [
            label, "overlay_multiplier", "ranked_probability",
            "temp", "temp_rating",
            "humidity", "humidity_rating",
            "wind_mph", "wind_rating",
            "wind_dir_string", "condition", "condition_rating"
        ]
        leaderboard = df[leaderboard_cols].copy()
        leaderboard[label] = leaderboard[label].round(4)
        leaderboard["ranked_probability"] = leaderboard["ranked_probability"].round(4)
        leaderboard["overlay_multiplier"] = leaderboard["overlay_multiplier"].round(3)
        return leaderboard

    # Choose which calibrated stream to display (default: Isotonic)
    leaderboard = build_leaderboard(today_df, today_iso, "hr_probability_iso")

    # ===== Outputs =====
    top_n = 30
    st.markdown(f"### 🏆 **Top {top_n} HR Leaderboard (Meta + Isotonic, ranked by logit(p)+log(overlay))**")
    leaderboard_top = leaderboard.head(top_n)
    st.dataframe(leaderboard_top, use_container_width=True)

    st.download_button(
        f"⬇️ Download Top {top_n} Leaderboard (Isotonic-ranked) CSV",
        data=leaderboard_top.to_csv(index=False),
        file_name=f"top{top_n}_leaderboard_iso_ranked.csv"
    )

    st.download_button(
        "⬇️ Download Full Prediction CSV (Isotonic-ranked)",
        data=leaderboard.to_csv(index=False),
        file_name="today_hr_predictions_full_iso_ranked.csv"
    )

    # Leaderboard plot
    if "player_name" in leaderboard_top.columns:
        st.subheader(f"📊 Ranked Probability Distribution (Top {top_n})")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(leaderboard_top["player_name"].astype(str), leaderboard_top["ranked_probability"])
        ax.invert_yaxis()
        ax.set_xlabel('Ranked HR Score (probability scale)')
        ax.set_ylabel('Player')
        st.pyplot(fig)
        plt.close(fig)

    # Drift diagnostics
    drifted = drift_check(X, X_today, n=6)
    if drifted:
        st.markdown("#### ⚡ **Feature Drift Diagnostics**")
        st.write("These features have unusual mean/std changes between training and today, check if input context shifted:", drifted)

    # Prediction histogram
    st.subheader("Prediction Probability Distribution (all predictions, Isotonic-ranked)")
    plt.figure(figsize=(8, 3))
    plt.hist(leaderboard["ranked_probability"], bins=30, alpha=0.7)
    plt.xlabel("Ranked HR Probability")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.close()

    # Memory cleanup
    del X, X_today, y, P_xgb_oof, P_lgb_oof, P_cat_oof, P_xgb_today, P_lgb_today, P_cat_today
    gc.collect()

else:
    st.warning("Upload both event-level and today CSVs (CSV or Parquet) to begin.")
