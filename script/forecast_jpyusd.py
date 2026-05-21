#!/usr/bin/env python3
"""
USD/JPY Exchange Rate Analysis and Forecasting with BISTRO.

Uses daily US and Japan 10-year bond yield spread (us_10y − jp_10y) together
with USD/JPY data from usdjpy_panel.csv.  Data are resampled to monthly before
Moirai inference.

Outputs
-------
script/figures/jpyusd_history.png      — historical time series (EDA)
script/figures/jpyusd_backtest.png     — rolling-window backtest vs AR(1)
script/figures/jpyusd_live_forecast.png — final live forecast for next month

Usage
-----
    python script/forecast_jpyusd.py          # full run (requires model weights)
    python script/forecast_jpyusd.py --test   # data pipeline only (no model)
"""

import sys
import argparse
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
SRC_ROOT   = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── configuration ──────────────────────────────────────────────────────────────
MODEL_REPO       = REPO_ROOT / "bistro-finetuned"
DATA_FILE        = REPO_ROOT / "data" / "usdjpy_panel.csv"
OUT_DIR          = SCRIPT_DIR / "figures"

FREQ             = "M"       # monthly data
PDT              = 12        # backtest forecast horizon (months)
CTX              = 120       # context window (10 years)
PSZ              = 32        # patch size in days (≈ 1 calendar month)
BSZ              = 32        # batch size for inference
ROLLING_WINDOWS  = 3         # number of backtesting windows
WINDOW_DISTANCE  = 2         # months between consecutive windows
FORECAST_START   = "2024-01-01"   # first backtest window start


# ── helpers ────────────────────────────────────────────────────────────────────
def _rmse(yhat, y):
    err = np.asarray(yhat, float) - np.asarray(y, float)
    return float(np.sqrt(np.nanmean(err ** 2)))


def _run_one_window(w, forecasts, inputs, labels, prep_monthly, target_col, pdt, psz, freq,
                    forecast_start, window_distance, ctx):
    """Aggregate samples for window w → monthly DataFrame + RMSE row."""
    from preprocessing_util import aggregate_daily_forecast_to_monthly
    from inference_util import ar1_forecast

    samples      = np.asarray(forecasts[w].samples, dtype=float)
    label_target = np.asarray(labels[w]["target"],  dtype=float)
    inp_target   = np.asarray(inputs[w]["target"],  dtype=float)
    last_input   = float(inp_target[-1]) if inp_target.size > 0 else None

    preds, _, ci = aggregate_daily_forecast_to_monthly(
        samples, label_target, last_input,
        steps_per_period=psz, expected_periods=pdt,
    )

    pred_index = pd.period_range(
        start=forecast_start + w * window_distance,
        periods=pdt, freq=freq,
    )

    dfw = pd.DataFrame(
        {"bistro_pred": preds, "bistro_lo": ci[:, 0], "bistro_hi": ci[:, 1]},
        index=pred_index,
    )

    # AR(1) baseline
    train_end_w = pred_index[0] - 1
    train_y = prep_monthly[target_col].loc[:train_end_w].tail(ctx).astype(float)
    try:
        ar1_pred = ar1_forecast(
            train_y, pred_index, method="statsmodels", trend="c", validate_index=True,
        )
    except Exception:
        ar1_pred = pd.Series(np.nan, index=pred_index)
    dfw["ar1_pred"] = ar1_pred

    actual  = prep_monthly[target_col].reindex(pred_index).astype(float)
    valid_b = actual.notna() & dfw["bistro_pred"].notna()
    valid_a = actual.notna() & ar1_pred.notna()

    rmse_b  = _rmse(dfw["bistro_pred"][valid_b], actual[valid_b]) if valid_b.any() else float("nan")
    rmse_a  = _rmse(ar1_pred[valid_a],           actual[valid_a]) if valid_a.any() else float("nan")
    r_rmse  = round(rmse_b / rmse_a, 4) if (rmse_b == rmse_b and rmse_a == rmse_a and rmse_a != 0) else float("nan")

    rmse_row = {
        "window":      w,
        "test_start":  pred_index[0],
        "test_end":    pred_index[-1],
        "rmse_bistro": round(rmse_b, 4),
        "rmse_ar1":    round(rmse_a, 4),
        "r_rmse":      r_rmse,
        "n_valid":     int(valid_b.sum()),
    }
    return dfw, rmse_row


# ── main ───────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and resample to monthly ──────────────────────────────────────
    print("=" * 60)
    print("USD/JPY Analysis with BISTRO")
    print("=" * 60)
    print(f"\n[1/6] Loading {DATA_FILE.name} ...")

    df_raw = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df_raw.index = pd.to_datetime(df_raw.index)

    # Keep only columns needed for this analysis
    cols_needed = ["usdjpy", "us_10y", "jp_10y", "spread", "vix"]
    df_sub = df_raw[cols_needed].dropna(subset=["usdjpy", "us_10y", "jp_10y"])

    # Resample: last trading day per month → convert to PeriodIndex
    # Use "M" (pandas 2.1 compatible); "ME" requires pandas 2.2+
    df_m = df_sub.resample("M").last()
    df_m.index = df_m.index.to_period("M")
    df_m = df_m.dropna(subset=["usdjpy"])

    target_col = "usdjpy"
    last_month  = df_m.index[-1]
    next_month  = last_month + 1

    print(f"  Monthly rows  : {len(df_m)}")
    print(f"  Date range    : {df_m.index[0]} → {last_month}")
    print(f"  Last USD/JPY  : {df_m[target_col].iloc[-1]:.2f}")
    print(f"  Forecast target: {next_month} (one month ahead)")

    # Monthly log-return stats
    log_ret = np.log(df_m[target_col] / df_m[target_col].shift(1)) * 100
    print(f"\n  Monthly log-return stats (%):")
    print(f"    mean={log_ret.mean():.3f}  std={log_ret.std():.3f}  "
          f"ann.vol={log_ret.std() * 12**0.5:.2f}")

    # ── 2. EDA plot ───────────────────────────────────────────────────────────
    print("\n[2/6] Saving historical EDA plot ...")

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    x = df_m.index.to_timestamp()

    axes[0].plot(x, df_m["usdjpy"],  color="black", lw=1.4)
    axes[0].set_ylabel("USD/JPY")
    axes[0].set_title("USD/JPY Exchange Rate — Historical Analysis")
    axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(x, df_m["spread"], color="C1", lw=1.2)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_ylabel("US−JP 10y Spread (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].plot(x, df_m["vix"], color="C3", lw=1.0, alpha=0.85)
    axes[2].set_ylabel("VIX")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)
    axes[2].spines[["top", "right"]].set_visible(False)

    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "jpyusd_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: jpyusd_history.png")

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    print(f"\n[3/6] Preparing data for BISTRO ...")

    from fx_preprocessing_util import (
        prepare_fx_monthly_for_inference,
        prepare_live_forecast,
    )
    from preprocessing_util import prepare_long_df_monthly_for_daily_inference

    # Univariate
    prep = prepare_fx_monthly_for_inference(
        df_m,
        target_col=target_col,
        freq=FREQ,
        use_log_returns=False,
        forecast_start_date=FORECAST_START,
        pdt_patches=PDT,
        ctx_patches=CTX,
        steps_per_period=PSZ,
        rolling_windows=ROLLING_WINDOWS,
        window_distance_patches=WINDOW_DISTANCE,
    )
    print(f"  Univariate backtest windows: {prep.windows}")
    print(f"  PDT steps={prep.pdt_steps}, CTX steps={prep.ctx_steps}")

    # Multivariate (with US−JP 10y spread as covariate)
    df_mv = df_m[[target_col]].rename(columns={target_col: "target"}).copy()
    df_mv["item_id"] = "USDJPY"
    df_mv = df_mv.join(df_m[["spread"]], how="inner")

    prep_mv = prepare_long_df_monthly_for_daily_inference(
        df_mv,
        item_id_col="item_id",
        target_col="target",
        past_dynamic_real_cols=["spread"],
        freq=FREQ,
        forecast_start_date=FORECAST_START,
        pdt_patches=PDT,
        ctx_patches=CTX,
        steps_per_period=PSZ,
        rolling_windows=ROLLING_WINDOWS,
        window_distance_patches=WINDOW_DISTANCE,
    )
    print(f"  Multivariate backtest windows: {prep_mv.windows}")

    # Live forecast data (univariate, next month)
    daily_live, cutoff_live, ctx_live = prepare_live_forecast(
        df_m[[target_col]],
        target_col=target_col,
        freq=FREQ,
        ctx_patches=CTX,
        steps_per_period=PSZ,
    )
    print(f"  Live forecast target: {next_month}")

    if dry_run:
        print("\n[--test mode] Data pipeline OK. Skipping model inference.")
        print("  To run full inference, execute without --test flag in an")
        print("  environment with uni2ts and model weights installed.")
        _save_test_plot(df_m, target_col, OUT_DIR)
        return

    # ── 4. Model inference ────────────────────────────────────────────────────
    print("\n[4/6] Running BISTRO inference ...")

    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    def _make_model(pdt_steps, ctx_steps, past_dim=0):
        return MoiraiForecast(
            module=MoiraiModule.from_pretrained(str(MODEL_REPO)),
            prediction_length=pdt_steps,
            context_length=ctx_steps,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=past_dim,
        )

    # ── Univariate backtest
    ds_univ = PandasDataset(prep.daily_df, target=target_col)
    _, test_tmpl = split(ds_univ, date=prep.cutoff_period_daily)
    test_data = test_tmpl.generate_instances(
        prediction_length=prep.pdt_steps,
        windows=prep.windows,
        distance=prep.dist_steps,
        max_history=prep.ctx_steps,
    )

    model_univ = _make_model(prep.pdt_steps, prep.ctx_steps, past_dim=0)
    predictor  = model_univ.create_predictor(batch_size=BSZ)
    inputs     = list(test_data.input)
    labels     = list(test_data.label)
    forecasts  = list(predictor.predict(test_data.input))

    bistro_univ = {}
    rmse_univ   = []
    for w in range(prep.windows):
        dfw, row = _run_one_window(
            w, forecasts, inputs, labels,
            prep.df_monthly, target_col, PDT, PSZ, FREQ,
            prep.forecast_start, WINDOW_DISTANCE, CTX,
        )
        bistro_univ[w] = dfw
        rmse_univ.append(row)

    # ── Multivariate backtest
    from gluonts.dataset.pandas import PandasDataset as PDS
    ds_mv = PDS.from_long_dataframe(
        prep_mv.daily_long_df,
        item_id="item_id",
        past_feat_dynamic_real=["spread"],
        feat_dynamic_real=[],
    )
    _, test_tmpl_mv = split(ds_mv, date=prep_mv.cutoff_period_daily)
    test_data_mv = test_tmpl_mv.generate_instances(
        prediction_length=prep_mv.pdt_steps,
        windows=prep_mv.windows,
        distance=prep_mv.dist_steps,
        max_history=prep_mv.ctx_steps,
    )

    model_mv = _make_model(
        prep_mv.pdt_steps, prep_mv.ctx_steps,
        past_dim=ds_mv.num_past_feat_dynamic_real,
    )
    predictor_mv = model_mv.create_predictor(batch_size=BSZ)
    inputs_mv    = list(test_data_mv.input)
    labels_mv    = list(test_data_mv.label)
    forecasts_mv = list(predictor_mv.predict(test_data_mv.input))

    bistro_mv   = {}
    rmse_mv_lst = []
    for w in range(prep_mv.windows):
        from preprocessing_util import aggregate_daily_forecast_to_monthly
        from inference_util import ar1_forecast

        samples      = np.asarray(forecasts_mv[w].samples, dtype=float)
        label_target = np.asarray(labels_mv[w]["target"],  dtype=float)
        inp_target   = np.asarray(inputs_mv[w]["target"],  dtype=float)
        last_input   = float(inp_target[-1]) if inp_target.size > 0 else None

        preds, _, ci = aggregate_daily_forecast_to_monthly(
            samples, label_target, last_input,
            steps_per_period=PSZ, expected_periods=PDT,
        )
        pred_index = pd.period_range(
            start=prep_mv.forecast_start + w * WINDOW_DISTANCE,
            periods=PDT, freq=FREQ,
        )
        dfw_mv = pd.DataFrame(
            {"bistro_mv_pred": preds, "bistro_mv_lo": ci[:, 0], "bistro_mv_hi": ci[:, 1]},
            index=pred_index,
        )

        train_end_w = pred_index[0] - 1
        train_y = prep_mv.df_monthly_target["target"].loc[:train_end_w].tail(CTX).astype(float)
        try:
            ar1_mv = ar1_forecast(train_y, pred_index, method="statsmodels", trend="c", validate_index=True)
        except Exception:
            ar1_mv = pd.Series(np.nan, index=pred_index)
        dfw_mv["ar1_pred"] = ar1_mv
        bistro_mv[w] = dfw_mv

        actual  = prep_mv.df_monthly_target["target"].reindex(pred_index).astype(float)
        valid_b = actual.notna() & dfw_mv["bistro_mv_pred"].notna()
        valid_a = actual.notna() & ar1_mv.notna()
        rmse_b  = _rmse(dfw_mv["bistro_mv_pred"][valid_b], actual[valid_b]) if valid_b.any() else float("nan")
        rmse_a  = _rmse(ar1_mv[valid_a], actual[valid_a]) if valid_a.any() else float("nan")
        r_rmse  = round(rmse_b / rmse_a, 4) if (rmse_b == rmse_b and rmse_a == rmse_a and rmse_a != 0) else float("nan")
        rmse_mv_lst.append({
            "window": w, "test_start": pred_index[0], "test_end": pred_index[-1],
            "rmse_bistro_mv": round(rmse_b, 4), "rmse_ar1": round(rmse_a, 4), "r_rmse": r_rmse,
        })

    # ── Live forecast (univariate, reuse univariate predictor)
    print(f"\n[5/6] Running live forecast for {next_month} ...")

    from preprocessing_util import aggregate_daily_forecast_to_monthly as _agg

    ds_live = PandasDataset(daily_live, target=target_col)
    _, test_live_tmpl = split(ds_live, date=cutoff_live)
    test_data_live = test_live_tmpl.generate_instances(
        prediction_length=PSZ,      # 1 month = 32 days
        windows=1,
        distance=PSZ,
        max_history=ctx_live,
    )

    model_live = _make_model(PSZ, ctx_live, past_dim=0)
    pred_live  = model_live.create_predictor(batch_size=BSZ)
    inputs_live   = list(test_data_live.input)
    labels_live   = list(test_data_live.label)
    forecasts_live = list(pred_live.predict(test_data_live.input))

    samples_live  = np.asarray(forecasts_live[0].samples, dtype=float)
    label_live    = np.asarray(labels_live[0]["target"],  dtype=float)
    inp_live      = np.asarray(inputs_live[0]["target"],  dtype=float)
    last_inp_live = float(inp_live[-1]) if inp_live.size > 0 else None

    preds_live, _, ci_live = _agg(
        samples_live, label_live, last_inp_live,
        steps_per_period=PSZ, expected_periods=1,
    )

    live_val = preds_live[0]
    live_lo  = ci_live[0, 0]
    live_hi  = ci_live[0, 1]

    # ── 5. Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — Univariate BISTRO")
    print("=" * 60)
    df_rmse_univ = pd.DataFrame(rmse_univ)
    print(df_rmse_univ.to_string(index=False))

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — Multivariate BISTRO (+ US-JP Spread)")
    print("=" * 60)
    df_rmse_mv = pd.DataFrame(rmse_mv_lst)
    print(df_rmse_mv.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"LIVE FORECAST — {next_month}")
    print("=" * 60)
    print(f"  Last observed ({last_month}): {df_m[target_col].iloc[-1]:.2f} USD/JPY")
    print(f"  Forecast median          : {live_val:.2f} USD/JPY")
    print(f"  90 % confidence interval : [{live_lo:.2f}, {live_hi:.2f}]")
    chg = live_val - df_m[target_col].iloc[-1]
    print(f"  Expected change          : {chg:+.2f} ({chg/df_m[target_col].iloc[-1]*100:+.2f}%)")
    print("=" * 60)

    # ── 6. Final plots ────────────────────────────────────────────────────────
    print("\n[6/6] Saving forecast plots ...")

    from inference_util import plot_publication_forecast_comparison

    # ── Backtest plot (window 0, both models)
    w = 0
    fc_start_w   = prep.forecast_start + w * WINDOW_DISTANCE
    plot_from_bt = fc_start_w - min(CTX, 48)
    plot_to_bt   = bistro_univ[w].index.max()

    df_actual_bt = prep.df_monthly[[target_col]].rename(columns={target_col: "actual"})
    df_plot_bt   = df_actual_bt.join(
        bistro_univ[w][["bistro_pred", "ar1_pred"]], how="outer"
    )
    if w in bistro_mv:
        df_plot_bt = df_plot_bt.join(bistro_mv[w][["bistro_mv_pred"]], how="outer")
    df_plot_bt = df_plot_bt.sort_index().loc[plot_from_bt:plot_to_bt]

    fc_cols = {"bistro_pred": "BISTRO Univariate", "ar1_pred": "AR(1)"}
    if "bistro_mv_pred" in df_plot_bt.columns:
        fc_cols["bistro_mv_pred"] = "BISTRO + Spread"

    fig_bt, ax_bt = plot_publication_forecast_comparison(
        df_plot_bt,
        actual_col="actual",
        forecast_cols=fc_cols,
        forecast_start=fc_start_w,
        title=f"USD/JPY — Backtest (window 0, start {fc_start_w})",
        ylabel="USD/JPY (yen per dollar)",
        savepaths=[OUT_DIR / "jpyusd_backtest.png"],
    )
    plt.close(fig_bt)
    print("  Saved: jpyusd_backtest.png")

    # ── Live forecast plot
    plot_from_lv = last_month - min(CTX, 60)
    df_actual_lv = df_m[[target_col]].rename(columns={target_col: "actual"})
    df_live_pt   = pd.DataFrame(
        {"live_pred": [live_val], "live_lo": [live_lo], "live_hi": [live_hi]},
        index=[next_month],
    )
    df_plot_lv = df_actual_lv.join(df_live_pt[["live_pred"]], how="outer")
    df_plot_lv = df_plot_lv.sort_index().loc[plot_from_lv:]

    fig_lv, ax_lv = plot_publication_forecast_comparison(
        df_plot_lv,
        actual_col="actual",
        forecast_cols={"live_pred": f"Forecast {next_month} (median)"},
        forecast_start=next_month,
        title=f"USD/JPY — Live Forecast for {next_month}",
        ylabel="USD/JPY (yen per dollar)",
        savepaths=[OUT_DIR / "jpyusd_live_forecast.png"],
    )

    # Add shaded CI band on live forecast
    ci_x = [next_month.to_timestamp()]
    ax_lv.fill_between(ci_x, [live_lo], [live_hi], color="C0", alpha=0.25, label="90% CI")
    ax_lv.legend(loc="upper left", frameon=False)
    fig_lv.savefig(OUT_DIR / "jpyusd_live_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig_lv)
    print("  Saved: jpyusd_live_forecast.png")

    print("\nAll done.")


def _save_test_plot(df_m, target_col, out_dir):
    """Light plot saved during --test run (no model required)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = df_m.index.to_timestamp()
    ax.plot(x, df_m[target_col], color="black", lw=1.2)
    ax.set_title("USD/JPY Monthly (test mode — model not loaded)")
    ax.set_ylabel("USD/JPY")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "jpyusd_test_run.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USD/JPY forecasting with BISTRO")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Data pipeline test only (skip model inference)",
    )
    args = parser.parse_args()
    main(dry_run=args.test)
