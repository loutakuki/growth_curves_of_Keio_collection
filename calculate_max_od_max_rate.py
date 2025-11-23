# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# ========== 配置区域 ==========
INPUT_XLSX = r"D:\YING LAB\劳\投稿论文\scientific data\curves2.xlsx"
SHEET_NAME = 'LB'
OUTPUT_XLSX = r"D:\YING LAB\劳\投稿论文\scientific data\LB_parameter2.xlsx"

# 平滑窗口改为 5
ROLLING_WINDOW = 5
ALLOW_SINGLE_POINT_MAXOD = True
# =================================


def ensure_time_index(df, time_col=None):
    # 使用首列作为时间列
    if time_col is None:
        time_col = df.columns[0]
    if time_col not in df.columns:
        raise ValueError(f"找不到时间列 '{time_col}'")
    df = df.dropna(subset=[time_col]).copy()
    df.set_index(time_col, inplace=True)
    return df


def time_index_to_numeric_hours(index):
    # 将时间索引转为 float 小时
    try:
        return pd.to_numeric(index, errors='raise').astype(float)
    except Exception:
        pass
    try:
        dt = pd.to_datetime(index)
        ns = dt.view('int64')
        return ns / 1e9 / 3600.0
    except Exception:
        arr = pd.to_numeric(index, errors='coerce')
        if np.isnan(arr).all():
            raise ValueError(f"无法识别时间索引，前5个索引: {list(index[:5])}")
        return arr.values.astype(float)


def smooth_series(series, window=5):
    return series.rolling(window, center=True, min_periods=1).mean()


def getMaxOD(series, window=5):
    s_ave = smooth_series(series, window=window)
    idx = s_ave.idxmax()
    return idx, float(s_ave.max())


def getMaxGradient(series, window=5):
    x_hours = time_index_to_numeric_hours(series.index)
    y_log = np.log(series.values)
    gradients = np.gradient(y_log, x_hours)
    g_smoothed = smooth_series(pd.Series(gradients, index=series.index), window=window)
    idx = g_smoothed.idxmax()
    return idx, float(g_smoothed.max())


def calculate_growth_parameters(df, rolling_window=5, allow_single_point=False):
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
                "Reason": "no positive values",
                "TotalPoints": total_points,
                "PositivePoints": pos_points
            })
            continue

        try:
            if pos_points >= 2:
                # Max OD
                max_od_time, max_od = getMaxOD(positive, window=rolling_window)

                # MaxOD 是否落在最后 rolling_window 个点
                pos_index = list(positive.index)
                max_pos = pos_index.index(max_od_time)
                if max_pos >= len(pos_index) - rolling_window:
                    OD_tail_flag = 1
                else:
                    OD_tail_flag = 0

                # Max Rate
                max_rate_time, max_rate = getMaxGradient(positive, window=rolling_window)
                Rate_flag = 2 if max_rate > 3 else 0

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
                # 只有1个有效点
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
                        "Reason": "not enough points",
                        "TotalPoints": total_points,
                        "PositivePoints": pos_points
                    })

        except Exception as e:
            excluded.append({
                "Sample": col,
                "Reason": f"exception: {str(e)}",
                "TotalPoints": total_points,
                "PositivePoints": pos_points
            })
            continue

    return pd.DataFrame(results), pd.DataFrame(excluded)


def extract_sample_number(name):
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

    # 排序
    if not metrics_df.empty:
        nums = metrics_df['Sample'].map(extract_sample_number)
        max_num = nums.max() if pd.notnull(nums).any() else 0
        nums = nums.fillna(max_num + 1).astype(int)
        metrics_df = metrics_df.assign(
            _SampleNum=nums
        ).sort_values('_SampleNum').drop(columns=['_SampleNum'])

    # 输出 Excel
    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        metrics_df.to_excel(writer, index=False, sheet_name='metrics')
        if not excluded_df.empty:
            excluded_df.to_excel(writer, index=False, sheet_name='excluded')

    # 控制台信息
    processed = set(metrics_df['Sample']) if not metrics_df.empty else set()
    missing = sorted(list(set(expected_samples) - processed))

    print(f"总列数: {len(expected_samples)}")
    print(f"成功计算: {len(processed)}")
    print(f"被排除: {len(missing)}")
    if missing:
        print("被排除样本: ", ", ".join(missing))
    print("结果已写入: ", OUTPUT_XLSX)


if __name__ == "__main__":
    main()
