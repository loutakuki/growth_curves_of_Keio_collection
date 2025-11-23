# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# ========== CONFIGURATION ==========
INPUT_XLSX = r"curves.xlsx"
SHEET_NAME = 'LB'
OUTPUT_XLSX = "LB_parameter.xlsx"

ROLLING_WINDOW = 5
ALLOW_SINGLE_POINT_MAXOD = True
# ====================================


def ensure_time_index(df, time_col=None):
    """Ensure the first column is used as time index."""
    if time_col is None:
        time_col = df.columns[0]
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found.")

    df = df.dropna(subset=[time_col]).copy()
    df.set_index(time_col, inplace=True)
    return df


def time_index_to_numeric_hours(index):
    """Convert index values to numeric hours."""
    # Try direct numeric conversion
    try:
        return pd.to_numeric(index, errors='raise').astype(float)
    except Exception:
        pass

    # Try datetime conversion
    try:
        dt = pd.to_datetime(index)
        ns = dt.view('int64')
        return ns / 1e9 / 3600.0
    except Exception:
        pass

    # Try best-effort numeric fallback
    arr = pd.to_numeric(index, errors='coerce')
    if np.isnan(arr).all():
        raise ValueError(f"Unable to interpret time index, sample values: {list(index[:5])}")
    return arr.astype(float)


def smooth_series(series, window=5):
    """Apply centered rolling mean smoothing."""
    return series.rolling(window, center=True, min_periods=1).mean()


def getMaxOD(series, window=5):
    """Compute maximum OD using smoothed OD."""
    smoothed = smooth_series(series, window=window)
    max_idx = smoothed.idxmax()
    return max_idx, float(smoothed.max())


def getMaxGradient(series, window=5):
    """Compute maximum growth rate based on log(OD) slope."""
    x_hours = time_index_to_numeric_hours(series.index)
    y_log = np.log(series.values)

    gradients = np.gradient(y_log, x_hours)
    smoothed_grad = smooth_series(pd.Series(gradients, index=series.index), window)

    max_idx = smoothed_grad.idxmax()
    return max_idx, float(smoothed_grad.max())


def calculate_growth_parameters(df, rolling_window=5, allow_single_point=False):
    """Compute MaxOD and MaxRate for each growth curve."""
    results = []
    excluded = []

    for col in df.columns:
        raw = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        positive = raw[raw > 0].dropna()

        pos_points = len(positive)
        total_points = len(raw.dropna())

        if pos_points == 0:
            excluded.append({
                "Sample": col,
                "Reason": "No positive values",
                "TotalPoints": total_points,
                "PositivePoints": pos_points
            })
            continue

        try:
            if pos_points >= 2:
                # Max OD
                max_od_time, max_od = getMaxOD(positive, window=rolling_window)

                # Flag: MaxOD occurs at the tail end of curve
                pos_index_list = list(positive.index)
                max_od_pos = pos_index_list.index(max_od_time)
                OD_tail_flag = 1 if max_od_pos >= len(pos_index_list) - rolling_window else 0

                # Max Rate
                max_rate_time, max_rate = getMaxGradient(positive, window=rolling_window)

                # === NEW LOGIC ===
                # Rate_flag = 2 IF max_rate_time (hours) <= 1
                # No longer use "max_rate > 3" criterion
                max_rate_hour = float(time_index_to_numeric_hours([max_rate_time])[0])
                Rate_flag = 2 if max_rate_hour <= 1 else 0

                results.append({
                    "Sample": col,
                    "MaxOD": max_od,
                    "MaxOD Index (h)": max_od_time,
                    "OD_tail_flag": OD_tail_flag,
                    "MaxRate (1/h)": max_rate,
                    "MaxRate Index (h)": max_rate_time,
                    "Rate_flag": Rate_flag
                })

            else:
                # Only one valid point
                if allow_single_point:
                    max_od_time, max_od = getMaxOD(positive, window=rolling_window)
                    results.append({
                        "Sample": col,
                        "MaxOD": max_od,
                        "MaxOD Index (h)": max_od_time,
                        "OD_tail_flag": np.nan,
                        "MaxRate (1/h)": np.nan,
                        "MaxRate Index (h)": np.nan,
                        "Rate_flag": np.nan
                    })
                else:
                    excluded.append({
                        "Sample": col,
                        "Reason": "Not enough positive points",
                        "TotalPoints": total_points,
                        "PositivePoints": pos_points
                    })

        except Exception as e:
            excluded.append({
                "Sample": col,
                "Reason": f"Exception: {str(e)}",
                "TotalPoints": total_points,
                "PositivePoints": pos_points
            })
            continue

    return pd.DataFrame(results), pd.DataFrame(excluded)


def extract_sample_number(name):
    """Extract numeric ID from sample names for sorting."""
    m = re.search(r'(\d+)', str(name))
    return int(m.group(1)) if m else None


def main():
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    df = ensure_time_index(df)

    expected_samples = list(df.columns)

    metrics_df, excluded_df = calculate_growth_parameters(
        df,
        rolling_window=ROLLING_WINDOW,
        allow_single_point=ALLOW_SINGLE_POINT_MAXOD
    )

    # Sorting
    if not metrics_df.empty:
        nums = metrics_df["Sample"].map(extract_sample_number)
        max_num = nums.max() if pd.notnull(nums).any() else 0
        nums = nums.fillna(max_num + 1).astype(int)

        metrics_df = (
            metrics_df.assign(_SampleNum=nums)
            .sort_values("_SampleNum")
            .drop(columns=["_SampleNum"])
        )

    # Write Excel
    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        metrics_df.to_excel(writer, index=False, sheet_name='metrics')
        if not excluded_df.empty:
            excluded_df.to_excel(writer, index=False, sheet_name='excluded')

    # Console summary
    processed = set(metrics_df["Sample"]) if not metrics_df.empty else set()
    missing = sorted(list(set(expected_samples) - processed))

    print(f"Total samples: {len(expected_samples)}")
    print(f"Successfully processed: {len(processed)}")
    print(f"Excluded: {len(missing)}")
    if missing:
        print("Excluded samples:", ", ".join(missing))
    print("Results saved to:", OUTPUT_XLSX)


if __name__ == "__main__":
    main()
