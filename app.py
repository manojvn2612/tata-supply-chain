import logging
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import os
import importlib.util
import uuid
from typing import Optional
import dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("supplychain-backend")
dotenv.load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MODEL_DIR = os.getenv("MODEL_DIR", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. LLM features will not work.")

app = FastAPI(
    title="Supply Chain AI",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, pd.DataFrame] = {}

# ─── Imports ──────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import base64
from io import BytesIO
from langchain.tools import tool


# ─── Graph tool (detailed, all cases) ────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return result


@tool
def graph_tool(input_str: str) -> str:
    """
    Generates a detailed base64-encoded PNG graph from JSON data.
    Input must be a JSON string of the data to visualize.
    Automatically detects column types and produces the most informative chart.
    """
    try:
        from io import StringIO
        df = pd.read_json(StringIO(input_str))

        PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2"]
        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.facecolor": "white",
        })

        # ── Case 1: Demand Forecast ──────────────────────────────────────────
        if "Predicted Demand" in df.columns and "Material" in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("Demand Forecast Analysis", fontsize=15, fontweight="bold", y=1.01)

            # 1a: Bar chart — demand per material
            ax = axes[0, 0]
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
            bars = ax.bar(range(len(df)), df["Predicted Demand"], color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["Material"].astype(str), rotation=45, ha="right", fontsize=8)
            ax.set_title("Predicted Demand by Material", fontweight="bold")
            ax.set_ylabel("Units")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

            # 1b: Inventory position (Stock vs ROP if available)
            ax = axes[0, 1]
            if "Stock" in df.columns and "ROP" in df.columns:
                x = range(len(df))
                ax.bar([i - 0.2 for i in x], df["Stock"], width=0.4, label="Current Stock", color=PALETTE[1], alpha=0.8)
                ax.bar([i + 0.2 for i in x], df["ROP"], width=0.4, label="Reorder Point", color=PALETTE[2], alpha=0.8)
                ax.set_xticks(list(x))
                ax.set_xticklabels(df["Material"].astype(str), rotation=45, ha="right", fontsize=8)
                ax.set_title("Stock vs Reorder Point", fontweight="bold")
                ax.set_ylabel("Units")
                ax.legend(fontsize=8)
            elif "Safety Stock" in df.columns:
                ax.bar(range(len(df)), df["Safety Stock"], color=PALETTE[3], alpha=0.8, edgecolor="white")
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df["Material"].astype(str), rotation=45, ha="right", fontsize=8)
                ax.set_title("Safety Stock by Material", fontweight="bold")
                ax.set_ylabel("Units")
            else:
                ax.axis("off")

            # 1c: Decision breakdown (REORDER vs NO ACTION)
            ax = axes[1, 0]
            if "Decision" in df.columns:
                decision_counts = df["Decision"].value_counts()
                wedge_colors = [PALETTE[2] if "REORDER" in k else PALETTE[1] for k in decision_counts.index]
                wedges, texts, autotexts = ax.pie(
                    decision_counts.values, labels=decision_counts.index,
                    autopct="%1.0f%%", colors=wedge_colors,
                    startangle=90, textprops={"fontsize": 9}
                )
                for at in autotexts:
                    at.set_fontsize(9)
                    at.set_fontweight("bold")
                ax.set_title("Reorder Decision Breakdown", fontweight="bold")
            else:
                ax.axis("off")

            # 1d: Reorder Qty
            ax = axes[1, 1]
            if "Reorder Qty" in df.columns:
                reorder_df = df[df["Reorder Qty"] > 0].copy()
                if not reorder_df.empty:
                    ax.barh(reorder_df["Material"].astype(str), reorder_df["Reorder Qty"],
                            color=PALETTE[0], alpha=0.85, edgecolor="white")
                    ax.set_title("Recommended Reorder Quantities", fontweight="bold")
                    ax.set_xlabel("Units")
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                else:
                    ax.text(0.5, 0.5, "No reorders needed ✓", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12, color=PALETTE[1], fontweight="bold")
                    ax.axis("off")
            else:
                ax.axis("off")

            plt.tight_layout()
            return f"GRAPH_BASE64:{_fig_to_b64(fig)}"

        # ── Case 2: Stockout / Monte Carlo Risk ──────────────────────────────
        elif "Stockout Probability" in df.columns and "Material" in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("Stockout Risk Analysis (Monte Carlo)", fontsize=15, fontweight="bold")

            # 2a: Stockout probability per material
            ax = axes[0, 0]
            probs = df["Stockout Probability"].values * 100
            bar_colors = [PALETTE[2] if p > 50 else PALETTE[3] if p > 20 else PALETTE[1] for p in probs]
            bars = ax.bar(range(len(df)), probs, color=bar_colors, edgecolor="white")
            ax.axhline(50, color=PALETTE[2], linestyle="--", linewidth=1.2, label="50% threshold", alpha=0.7)
            ax.axhline(20, color=PALETTE[3], linestyle="--", linewidth=1.0, label="20% threshold", alpha=0.7)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["Material"].astype(str), rotation=45, ha="right", fontsize=8)
            ax.set_title("Stockout Probability by Material", fontweight="bold")
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 110)
            ax.legend(fontsize=8)
            for bar, p in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{p:.0f}%", ha="center", va="bottom", fontsize=7)

            # 2b: Risk distribution histogram
            ax = axes[0, 1]
            ax.hist(probs, bins=min(10, len(probs)), color=PALETTE[0], edgecolor="white",
                    alpha=0.85, rwidth=0.85)
            ax.axvline(probs.mean(), color=PALETTE[2], linestyle="--", linewidth=1.5,
                       label=f"Mean: {probs.mean():.1f}%")
            ax.set_title("Risk Distribution Histogram", fontweight="bold")
            ax.set_xlabel("Stockout Probability (%)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

            # 2c: Demand uncertainty band (if Mean/5%/95% available)
            ax = axes[1, 0]
            if "Mean Demand" in df.columns and "5% Demand" in df.columns and "95% Demand" in df.columns:
                x = range(len(df))
                ax.fill_between(x, df["5% Demand"], df["95% Demand"],
                                alpha=0.25, color=PALETTE[0], label="5%–95% band")
                ax.plot(x, df["Mean Demand"], color=PALETTE[0], linewidth=2, marker="o",
                        markersize=5, label="Mean Demand")
                ax.plot(x, df["Predicted Demand"] if "Predicted Demand" in df.columns else df["Mean Demand"],
                        color=PALETTE[2], linewidth=1.5, linestyle="--", marker="s",
                        markersize=4, label="Predicted")
                ax.set_xticks(list(x))
                ax.set_xticklabels(df["Material"].astype(str), rotation=45, ha="right", fontsize=8)
                ax.set_title("Demand Uncertainty Band", fontweight="bold")
                ax.set_ylabel("Units")
                ax.legend(fontsize=8)
            else:
                ax.axis("off")

            # 2d: Risk category pie
            ax = axes[1, 1]
            high = int((probs > 50).sum())
            med = int(((probs > 20) & (probs <= 50)).sum())
            low = int((probs <= 20).sum())
            if high + med + low > 0:
                wedges, texts, autotexts = ax.pie(
                    [high, med, low],
                    labels=[f"High Risk\n({high})", f"Medium Risk\n({med})", f"Low Risk\n({low})"],
                    colors=[PALETTE[2], PALETTE[3], PALETTE[1]],
                    autopct="%1.0f%%", startangle=90,
                    textprops={"fontsize": 9}
                )
                for at in autotexts:
                    at.set_fontweight("bold")
            ax.set_title("Risk Category Distribution", fontweight="bold")

            plt.tight_layout()
            return f"GRAPH_BASE64:{_fig_to_b64(fig)}"

        # ── Case 3: Supplier Risk (Cluster-based) ────────────────────────────
        elif "Supplier_Risk" in df.columns or "Supplier Risk" in df.columns:
            risk_col = "Supplier_Risk" if "Supplier_Risk" in df.columns else "Supplier Risk"
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("Supplier Risk Analysis", fontsize=15, fontweight="bold")

            risk_color_map = {"Low Risk": PALETTE[1], "Medium Risk": PALETTE[3], "High Risk": PALETTE[2]}

            # 3a: Risk distribution pie
            ax = axes[0, 0]
            counts = df[risk_col].value_counts()
            colors_pie = [risk_color_map.get(k, PALETTE[0]) for k in counts.index]
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index,
                autopct="%1.0f%%", colors=colors_pie,
                startangle=90, textprops={"fontsize": 9}
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax.set_title("Supplier Risk Distribution", fontweight="bold")

            # 3b: Risk by material bar chart
            ax = axes[0, 1]
            if "Material" in df.columns:
                bar_colors = [risk_color_map.get(r, PALETTE[0]) for r in df[risk_col]]
                ax.barh(df["Material"].astype(str), [1] * len(df),
                        color=bar_colors, edgecolor="white")
                ax.set_title("Risk Category per Material", fontweight="bold")
                ax.set_xlabel("")
                ax.set_xticks([])
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=v, label=k) for k, v in risk_color_map.items()]
                ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

            # 3c: Total_Time (lead time) distribution by risk
            ax = axes[1, 0]
            if "Total_Time" in df.columns:
                for risk_level, color in risk_color_map.items():
                    subset = df[df[risk_col] == risk_level]["Total_Time"].dropna()
                    if not subset.empty:
                        ax.hist(subset, bins=6, color=color, alpha=0.6,
                                label=risk_level, edgecolor="white")
                ax.set_title("Lead Time Distribution by Risk", fontweight="bold")
                ax.set_xlabel("Total Lead Time (Days)")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
            elif "Lead Time Supplier→Plant (Days)" in df.columns:
                for risk_level, color in risk_color_map.items():
                    subset = df[df[risk_col] == risk_level]["Lead Time Supplier→Plant (Days)"].dropna()
                    if not subset.empty:
                        ax.hist(subset, bins=6, color=color, alpha=0.6,
                                label=risk_level, edgecolor="white")
                ax.set_title("Lead Time Distribution by Risk", fontweight="bold")
                ax.set_xlabel("Lead Time (Days)")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
            else:
                ax.axis("off")

            # 3d: Count bar
            ax = axes[1, 1]
            risk_order = ["Low Risk", "Medium Risk", "High Risk"]
            risk_vals = [counts.get(r, 0) for r in risk_order]
            bar_colors2 = [risk_color_map[r] for r in risk_order]
            bars = ax.bar(risk_order, risk_vals, color=bar_colors2, edgecolor="white", width=0.5)
            ax.set_title("Materials per Risk Category", fontweight="bold")
            ax.set_ylabel("Count")
            for bar, val in zip(bars, risk_vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            str(val), ha="center", va="bottom", fontweight="bold")

            plt.tight_layout()
            return f"GRAPH_BASE64:{_fig_to_b64(fig)}"

        # ── Case 4: Policy Optimization ─────────────────────────────────────
        elif any(c in df.columns for c in ["total_cost", "Total Cost", "stockouts", "Total Stockouts"]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Policy Optimization Comparison", fontsize=14, fontweight="bold")

            cost_col = next((c for c in ["total_cost", "Total Cost"] if c in df.columns), None)
            so_col = next((c for c in ["stockouts", "Total Stockouts"] if c in df.columns), None)

            if cost_col:
                ax = axes[0]
                ax.bar(df.index if "Policy" not in df.columns else df["Policy"],
                       df[cost_col], color=[PALETTE[2], PALETTE[1]], edgecolor="white", width=0.5)
                ax.set_title("Total Cost Comparison", fontweight="bold")
                ax.set_ylabel("Cost ($)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

            if so_col:
                ax = axes[1]
                ax.bar(df.index if "Policy" not in df.columns else df["Policy"],
                       df[so_col], color=[PALETTE[2], PALETTE[1]], edgecolor="white", width=0.5)
                ax.set_title("Stockout Events Comparison", fontweight="bold")
                ax.set_ylabel("Events")

            plt.tight_layout()
            return f"GRAPH_BASE64:{_fig_to_b64(fig)}"

        # ── Fallback: generic numeric plot ──────────────────────────────────
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            label_col = next((c for c in df.columns if df[c].dtype == object), None)

            if not numeric_cols:
                return "No numeric data to plot"

            fig, ax = plt.subplots(figsize=(10, 5))
            for i, col in enumerate(numeric_cols[:4]):
                if label_col:
                    ax.bar([f"{str(v)[:8]}" for v in df[label_col]],
                           df[col], label=col, alpha=0.7,
                           color=PALETTE[i % len(PALETTE)])
                else:
                    ax.plot(df[col], label=col, marker="o",
                            color=PALETTE[i % len(PALETTE)], linewidth=2, markersize=4)
            ax.set_title("Data Overview", fontweight="bold")
            ax.legend(fontsize=9)
            if label_col:
                plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            return f"GRAPH_BASE64:{_fig_to_b64(fig)}"

    except Exception as e:
        logger.error(f"Graph tool error: {e}\n{traceback.format_exc()}")
        return f"Graph error: {str(e)}"


# ─── ML Tools ─────────────────────────────────────────────────────────────────

def import_from_path(module_name, file_path, func_name):
    if not os.path.exists(file_path):
        logger.error(f"Module file not found: {file_path}")
        raise ImportError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


try:
    run_lstm_demand_forecast = import_from_path(
        "lstm_train", os.path.join(MODEL_DIR, "lstm_train.py"), "run_lstm_demand_forecast"
    )
    run_montecarlo_risk = import_from_path(
        "montecarlo", os.path.join(MODEL_DIR, "montecarlo.py"), "run_montecarlo_risk"
    )
    run_supplier_risk = import_from_path(
        "supplier_risk", os.path.join(MODEL_DIR, "supplier_risk.py"), "run_supplier_risk"
    )
except Exception as e:
    logger.error(f"Error importing model modules: {e}")
    raise

from src.policy_optimizer import optimize_policy, format_policy_output


@tool
def demand_forecast_tool(data_json: str) -> str:
    """
    Runs LSTM model to forecast demand and returns demand predictions per material.
    Input: JSON string of the supply chain DataFrame.
    Returns: table with Material, Predicted Demand, Safety Stock, Stock, Open PO, ROP, Decision, Reorder Qty.
    Always call graph_tool after this with the same JSON to visualize results.
    """
    try:
        from io import StringIO
        df = pd.read_json(StringIO(data_json))
        results, _ = run_lstm_demand_forecast(df)
        return results.to_json(orient="records")
    except Exception as e:
        logger.error(f"demand_forecast_tool error: {e}")
        return f"Error: {str(e)}"


@tool
def risk_analysis_tool(data_json: str) -> str:
    """
    Performs Monte Carlo simulation to estimate stockout risk and uncertainty.
    Input: JSON string of the supply chain DataFrame.
    Returns: JSON with Material, Predicted Demand, Stockout Probability, etc.
    Always call graph_tool after this with the results JSON to visualize.
    """
    try:
        from io import StringIO
        df = pd.read_json(StringIO(data_json))
        results, _ = run_lstm_demand_forecast(df)
        if "Lead Time" not in results.columns:
            results["Lead Time"] = 10
        risk_df = run_montecarlo_risk(results)
        return risk_df.to_json(orient="records")
    except Exception as e:
        logger.error(f"risk_analysis_tool error: {e}")
        return f"Error: {str(e)}"


@tool
def supplier_risk_tool(data_json: str) -> str:
    """
    Clusters suppliers into risk categories (Low Risk, Medium Risk, High Risk).
    Input: JSON string of the supply chain DataFrame.
    Returns: JSON with Material, Supplier, Supplier_Risk columns and silhouette score.
    Always call graph_tool after this with the results JSON to visualize.
    """
    try:
        from io import StringIO
        df = pd.read_json(StringIO(data_json))
        risk_df, score = run_supplier_risk(df)
        result_df = risk_df[["Material", "Supplier_Risk", "Total_Time"]].copy() if "Material" in risk_df.columns else risk_df[["Supplier_Risk", "Total_Time"]].copy()
        return result_df.to_json(orient="records") + f"\nSilhouette Score: {score:.3f}"
    except Exception as e:
        logger.error(f"supplier_risk_tool error: {e}")
        return f"Error: {str(e)}"


@tool
def policy_optimization_tool(data_json: str) -> str:
    """
    Optimizes inventory policy using forecast, stockout risk, and supplier risk.
    Input: JSON string of the supply chain DataFrame.
    Returns: structured text comparing Naive vs Smart policy with cost savings.
    """
    try:
        from io import StringIO
        df = pd.read_json(StringIO(data_json))

        forecast_df, _ = run_lstm_demand_forecast(df)
        risk_df = run_montecarlo_risk(forecast_df)
        supplier_df, _ = run_supplier_risk(df)

        stockout_risk = float(risk_df["Stockout Probability"].mean())
        supplier_risk_mode = supplier_df["Supplier_Risk"].mode()[0]

        result = optimize_policy(
            data={"initial_stock": float(forecast_df["Stock"].mean()), "forecast": forecast_df["Predicted Demand"].tolist()},
            forecast=forecast_df["Predicted Demand"].tolist(),
            stockout_risk=stockout_risk,
            supplier_risk=supplier_risk_mode.replace(" Risk", "").upper(),
        )

        return format_policy_output(result)
    except Exception as e:
        logger.error(f"policy_optimization_tool error: {e}")
        return f"Error: {str(e)}"


# ─── Direct ML runner (bypasses LLM agent for clean results) ─────────────────

def _run_direct(df: pd.DataFrame, question: str) -> dict:
    """
    Direct (non-agent) ML runner. Detects intent and calls the right ML function.
    Returns {"answer": str, "raw_output": str, "graph_b64": str|None}
    """
    q = question.lower()
    graph_b64 = None
    raw_output = ""
    answer = ""

    try:
        if any(k in q for k in ["demand", "forecast", "lstm", "predict"]):
            results, _ = run_lstm_demand_forecast(df)
            # Use JSON so the frontend DataTable can parse rows correctly
            raw_output = results.to_json(orient="records")

            reorder_items = results[results["Decision"] == "REORDER"]
            answer = (
                f"**Demand Forecast Results**\n\n"
                f"Analyzed {len(results)} materials.\n"
                f"- **{len(reorder_items)}** materials require reordering.\n"
                f"- Average predicted demand: **{results['Predicted Demand'].mean():.1f} units**\n"
                f"- Highest demand: **{results['Predicted Demand'].max():.1f} units** "
                f"({results.loc[results['Predicted Demand'].idxmax(), 'Material']})\n\n"
            )
            if not reorder_items.empty:
                answer += "**Materials needing reorder:**\n"
                for _, row in reorder_items.iterrows():
                    answer += f"- {row['Material']}: reorder {row['Reorder Qty']:.0f} units (ROP={row['ROP']:.1f})\n"

            graph_result = graph_tool.invoke(results.to_json(orient="records"))
            if graph_result.startswith("GRAPH_BASE64:"):
                graph_b64 = graph_result.split("GRAPH_BASE64:", 1)[1].strip()

        # ── Supplier branch MUST come before stockout: "supplier"/"cluster"/"vendor"
        # are exclusive keywords; "risk" alone would collide with stockout branch.
        elif any(k in q for k in ["supplier", "cluster", "vendor"]):
            risk_df, score = run_supplier_risk(df)
            # Return only the columns the frontend needs; use JSON for parseRows
            cols = ["Material", "Supplier_Risk", "Total_Time"] if "Material" in risk_df.columns else ["Supplier_Risk", "Total_Time"]
            raw_output = risk_df[cols].to_json(orient="records")

            counts = risk_df["Supplier_Risk"].value_counts()
            answer = (
                f"**Supplier Risk Analysis (K-Means Clustering)**\n\n"
                f"Silhouette score: **{score:.3f}** (higher = better clustering)\n\n"
            )
            for risk_level, count in counts.items():
                answer += f"- **{risk_level}**: {count} materials\n"

            graph_result = graph_tool.invoke(risk_df.to_json(orient="records"))
            if graph_result.startswith("GRAPH_BASE64:"):
                graph_b64 = graph_result.split("GRAPH_BASE64:", 1)[1].strip()

        elif any(k in q for k in ["stockout", "monte carlo", "simulation", "stockout risk"]):
            results, _ = run_lstm_demand_forecast(df)
            if "Lead Time" not in results.columns:
                results["Lead Time"] = 10
            risk_df = run_montecarlo_risk(results)
            # Use JSON so the frontend DataTable can parse rows correctly
            raw_output = risk_df.to_json(orient="records")

            high_risk = risk_df[risk_df["Stockout Probability"] > 0.5]
            avg_prob = risk_df["Stockout Probability"].mean()
            answer = (
                f"**Monte Carlo Stockout Risk Analysis**\n\n"
                f"Simulated 10,000 demand scenarios per material.\n"
                f"- Average stockout probability: **{avg_prob:.1%}**\n"
                f"- **{len(high_risk)}** materials have >50% stockout risk.\n\n"
            )
            if not high_risk.empty:
                answer += "**High-risk materials (>50% stockout probability):**\n"
                for _, row in high_risk.iterrows():
                    answer += f"- {row['Material']}: {row['Stockout Probability']:.1%} stockout probability\n"

            graph_result = graph_tool.invoke(risk_df.to_json(orient="records"))
            if graph_result.startswith("GRAPH_BASE64:"):
                graph_b64 = graph_result.split("GRAPH_BASE64:", 1)[1].strip()

        elif any(k in q for k in ["policy", "optim", "strategy", "inventory"]):
            forecast_df, _ = run_lstm_demand_forecast(df)
            risk_df = run_montecarlo_risk(forecast_df)
            supplier_df, _ = run_supplier_risk(df)

            stockout_risk = float(risk_df["Stockout Probability"].mean())
            supplier_risk_mode = supplier_df["Supplier_Risk"].mode()[0]

            result = optimize_policy(
                data={"initial_stock": float(forecast_df["Stock"].mean()), "forecast": forecast_df["Predicted Demand"].tolist()},
                forecast=forecast_df["Predicted Demand"].tolist(),
                stockout_risk=stockout_risk,
                supplier_risk=supplier_risk_mode.replace(" Risk", "").upper(),
            )
            raw_output = format_policy_output(result)

            n = result.naive_metrics
            s = result.smart_metrics
            cost_saved = result.cost_savings
            answer = (
                f"**Policy Optimization Complete**\n\n"
                f"Recommendation: **{result.recommended_policy}**\n\n"
                f"**Smart Policy Parameters:**\n"
                f"- Safety Stock: {result.safety_stock:.1f} units\n"
                f"- Reorder Point: {result.recommended_reorder_point:.1f} units\n"
                f"- Order Quantity: {result.recommended_order_quantity:.1f} units\n"
                f"- Lead Time: {result.lead_time} days\n"
                f"- Service Level: {result.service_level:.0%}\n\n"
                f"**Cost Comparison:**\n"
                f"- Naive Policy total cost: ${n.total_cost:,.2f} ({n.total_stockouts} stockouts)\n"
                f"- Smart Policy total cost: ${s.total_cost:,.2f} ({s.total_stockouts} stockouts)\n"
                f"- **Savings: ${abs(cost_saved):,.2f}** ({'saving' if cost_saved >= 0 else 'extra cost'})\n"
            )

            # Build comparison df for graph
            comp_df = pd.DataFrame({
                "Policy": ["Naive", "Smart"],
                "total_cost": [n.total_cost, s.total_cost],
                "Total Stockouts": [n.total_stockouts, s.total_stockouts],
            })
            graph_result = graph_tool.invoke(comp_df.to_json(orient="records"))
            if graph_result.startswith("GRAPH_BASE64:"):
                graph_b64 = graph_result.split("GRAPH_BASE64:", 1)[1].strip()

        else:
            # General question — use LLM
            return _llm_answer(df, question)

    except Exception as e:
        logger.error(f"Direct runner error: {e}\n{traceback.format_exc()}")
        answer = f"An error occurred while processing your request: {str(e)}"
        raw_output = traceback.format_exc()

    return {"answer": answer, "raw_output": raw_output, "graph": graph_b64}


def _llm_answer(df: pd.DataFrame, question: str) -> dict:
    """
    Use the LLM for general / non-ML questions. No agent loop — direct call.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if not OPENROUTER_API_KEY:
        return {"answer": "LLM not configured (OPENROUTER_API_KEY missing).", "raw_output": "", "graph": None}

    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3-8b-instruct",
        temperature=0,
        max_tokens=800,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful supply chain AI assistant.\n"
        "Dataset summary: {summary}\n\n"
        "User question: {question}\n\n"
        "Provide a clear, concise answer based on the dataset. Do not call any tools."
    )
    chain = prompt | llm | StrOutputParser()

    summary = f"{len(df)} rows, columns: {list(df.columns)[:10]}"
    answer = chain.invoke({"summary": summary, "question": question})

    return {"answer": answer, "raw_output": "", "graph": None}


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    raw_output: Optional[str] = None
    graph: Optional[str] = None


class DatasetMeta(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    unique_suppliers: Optional[int] = None
    unique_materials: Optional[int] = None
    preview: list[dict]


# ─── Exception handler ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=DatasetMeta)
async def upload_file(file: UploadFile = File(...)):
    """Accept an Excel file; return a session_id and dataset metadata."""
    if not file.filename.endswith(".xlsx"):
        logger.warning(f"Rejected upload: {file.filename}")
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Could not parse file: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = df

    meta = DatasetMeta(
        session_id=session_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        unique_suppliers=int(df["Supplier"].nunique()) if "Supplier" in df.columns else None,
        unique_materials=int(df["Material"].nunique()) if "Material" in df.columns else None,
        preview=df.head(8).fillna("").to_dict(orient="records"),
    )
    logger.info(f"Uploaded file: {file.filename} (session: {session_id})")
    return meta


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run direct ML pipeline and return answer + graph."""
    df = _sessions.get(req.session_id)
    if df is None:
        logger.warning(f"Session not found: {req.session_id}")
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    try:
        result = _run_direct(df, req.question)
        return ChatResponse(
            answer=result.get("answer", ""),
            raw_output=result.get("raw_output", ""),
            graph=result.get("graph"),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session."""
    _sessions.pop(session_id, None)
    logger.info(f"Session deleted: {session_id}")
    return {"deleted": session_id}