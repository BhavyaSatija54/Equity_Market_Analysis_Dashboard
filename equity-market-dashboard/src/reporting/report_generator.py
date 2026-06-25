"""
src/reporting/report_generator.py
-----------------------------------
Generates HTML, PDF, and Excel investment analytics reports
from aggregated analytics data.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Report data container
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    title: str
    as_of: date
    total_equities: int
    market_avg_return: float
    vol_regime: str
    sector_performance: pd.DataFrame      # columns: sector, avg_return, sharpe, n_equities
    top_performers: pd.DataFrame          # top 10 by 1Y return
    bottom_performers: pd.DataFrame       # bottom 10
    scenario_summary: dict[str, dict]     # bull/base/bear → {return, vol, sharpe}
    risk_metrics: dict[str, float]        # portfolio-level
    sections: list[str] = field(default_factory=lambda: [
        "market_summary", "top_performers", "risk_metrics",
        "scenario_analysis", "sector_breakdown"
    ])


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Arial, sans-serif;
          background:#0d1117; color:#e6edf3; margin:0; padding:32px; font-size:13px; }}
  h1   {{ color:#58a6ff; margin-bottom:4px; }}
  h2   {{ color:#e6edf3; border-bottom:1px solid #30363d; padding-bottom:6px; margin-top:32px; }}
  .meta {{ color:#8b949e; font-size:12px; margin-bottom:24px; }}
  .kpi-row {{ display:flex; gap:12px; flex-wrap:wrap; margin:20px 0; }}
  .kpi {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:14px 20px; min-width:160px; }}
  .kpi-label {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi-val   {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .pos {{ color:#3fb950; }} .neg {{ color:#f85149; }} .neutral {{ color:#58a6ff; }}
  table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:12px; }}
  th    {{ background:#161b22; color:#8b949e; padding:8px 12px; text-align:left;
           font-size:11px; text-transform:uppercase; letter-spacing:.4px; }}
  td    {{ padding:7px 12px; border-bottom:1px solid #21262d; }}
  tr:hover td {{ background:#161b22; }}
  .scenario-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0; }}
  .scen {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .scen-title {{ font-weight:700; margin-bottom:10px; }}
  .scen-metric {{ display:flex; justify-content:space-between; margin:5px 0; font-size:12px; }}
  footer {{ margin-top:48px; font-size:11px; color:#6e7681; border-top:1px solid #21262d;
            padding-top:16px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Generated: {as_of} &nbsp;|&nbsp; Universe: {n_equities} equities &nbsp;|&nbsp; Regime: {regime}</div>

{kpi_section}
{market_section}
{performers_section}
{scenario_section}
{sector_section}
{risk_section}

<footer>
  Equity Market Analysis Dashboard &bull; PySpark / Databricks Pipeline &bull; {as_of}
</footer>
</body>
</html>"""


def _pct(v: float, d: int = 2) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{d}f}%"


def _kpi_html(label: str, value: str, cls: str = "") -> str:
    return (
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val {cls}">{value}</div>'
        f'</div>'
    )


def _df_to_html(df: pd.DataFrame, cols: Optional[list[str]] = None) -> str:
    if cols:
        df = df[cols]
    return df.to_html(index=False, border=0, classes="")


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ReportGenerator:

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html(self, data: ReportData) -> str:
        """Generate HTML report and write to disk. Returns file path."""
        report_id = uuid.uuid4().hex[:8]
        filename  = f"equity_report_{data.as_of}_{report_id}.html"
        path      = self.output_dir / filename

        kpi_section = self._kpi_section(data)
        market_section    = "" if "market_summary"    not in data.sections else self._market_section(data)
        performers_section= "" if "top_performers"    not in data.sections else self._performers_section(data)
        scenario_section  = "" if "scenario_analysis" not in data.sections else self._scenario_section(data)
        sector_section    = "" if "sector_breakdown"  not in data.sections else self._sector_section(data)
        risk_section      = "" if "risk_metrics"      not in data.sections else self._risk_section(data)

        html = _HTML_TEMPLATE.format(
            title=data.title,
            as_of=str(data.as_of),
            n_equities=data.total_equities,
            regime=data.vol_regime,
            kpi_section=kpi_section,
            market_section=market_section,
            performers_section=performers_section,
            scenario_section=scenario_section,
            sector_section=sector_section,
            risk_section=risk_section,
        )

        path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report written -> {path}")
        return str(path)

    def generate_excel(self, data: ReportData) -> str:
        """Generate Excel workbook with one sheet per section."""
        report_id = uuid.uuid4().hex[:8]
        filename  = f"equity_report_{data.as_of}_{report_id}.xlsx"
        path      = self.output_dir / filename

        with pd.ExcelWriter(str(path), engine="xlsxwriter") as writer:
            wb = writer.book

            # Formats
            hdr_fmt = wb.add_format({"bold": True, "bg_color": "#0f3460", "font_color": "white"})
            pos_fmt = wb.add_format({"font_color": "#3fb950"})
            neg_fmt = wb.add_format({"font_color": "#f85149"})

            def write_sheet(df: pd.DataFrame, sheet: str) -> None:
                df.to_excel(writer, sheet_name=sheet, index=False)
                ws = writer.sheets[sheet]
                for col_idx, col in enumerate(df.columns):
                    ws.write(0, col_idx, col, hdr_fmt)
                ws.set_column(0, len(df.columns), 14)

            write_sheet(data.sector_performance,  "Sector Performance")
            write_sheet(data.top_performers,      "Top Performers")
            write_sheet(data.bottom_performers,   "Bottom Performers")

            # Scenario sheet
            scen_rows = [
                {"Scenario": k, **v}
                for k, v in data.scenario_summary.items()
            ]
            write_sheet(pd.DataFrame(scen_rows), "Scenarios")

            # Risk sheet
            risk_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in data.risk_metrics.items()])
            write_sheet(risk_df, "Risk Metrics")

        logger.info(f"Excel report written -> {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _kpi_section(self, data: ReportData) -> str:
        ret_cls = "pos" if data.market_avg_return >= 0 else "neg"
        kpis = [
            _kpi_html("Universe",       str(data.total_equities)),
            _kpi_html("Avg 1Y Return",  _pct(data.market_avg_return), ret_cls),
            _kpi_html("Vol Regime",     data.vol_regime, "neutral"),
            _kpi_html("Sharpe",         f'{data.risk_metrics.get("sharpe_ratio", 0):.2f}'),
            _kpi_html("VaR 95% (1D)",   _pct(data.risk_metrics.get("var_95", 0)), "neg"),
        ]
        return '<div class="kpi-row">' + "".join(kpis) + "</div>"

    def _market_section(self, data: ReportData) -> str:
        ret_cls = "pos" if data.market_avg_return >= 0 else "neg"
        return f"""<h2>Market Summary</h2>
<p>Equal-weight portfolio 1Y return: <span class="{ret_cls}">{_pct(data.market_avg_return)}</span>
&nbsp;|&nbsp; Volatility regime: <strong>{data.vol_regime}</strong></p>"""

    def _performers_section(self, data: ReportData) -> str:
        top_cols  = [c for c in ["ticker","sector","return_1y","vol_20d","beta","pe_ratio"] if c in data.top_performers.columns]
        bot_cols  = top_cols
        return (
            "<h2>Top Performers (1Y)</h2>"
            + _df_to_html(data.top_performers, top_cols)
            + "<h2>Bottom Performers (1Y)</h2>"
            + _df_to_html(data.bottom_performers, bot_cols)
        )

    def _scenario_section(self, data: ReportData) -> str:
        def scen_card(name: str, m: dict, cls: str) -> str:
            ret = m.get("portfolio_return", 0)
            ret_cls = "pos" if ret >= 0 else "neg"
            return (
                f'<div class="scen">'
                f'<div class="scen-title" style="color:{"#3fb950" if cls=="bull" else "#f85149" if cls=="bear" else "#58a6ff"}">'
                f'{name.upper()}</div>'
                f'<div class="scen-metric"><span>Portfolio Return</span>'
                f'<span class="{ret_cls}">{_pct(ret)}</span></div>'
                f'<div class="scen-metric"><span>Volatility</span>'
                f'<span>{_pct(m.get("portfolio_vol",0),1)}</span></div>'
                f'<div class="scen-metric"><span>Sharpe</span>'
                f'<span>{m.get("sharpe_ratio",0):.2f}</span></div>'
                f'<div class="scen-metric"><span>VaR 95%</span>'
                f'<span class="neg">{_pct(m.get("var_95",0))}</span></div>'
                f'</div>'
            )

        cards = "".join(
            scen_card(k, v, k)
            for k, v in data.scenario_summary.items()
        )
        return f'<h2>Scenario Analysis</h2><div class="scenario-grid">{cards}</div>'

    def _sector_section(self, data: ReportData) -> str:
        cols = [c for c in ["sector","n_equities","avg_return","annualised_volatility","sharpe_ratio","avg_beta"]
                if c in data.sector_performance.columns]
        return "<h2>Sector Breakdown</h2>" + _df_to_html(data.sector_performance, cols)

    def _risk_section(self, data: ReportData) -> str:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            for k, v in data.risk_metrics.items()
        )
        return (
            "<h2>Portfolio Risk Metrics</h2>"
            "<table><tr><th>Metric</th><th>Value</th></tr>"
            + rows + "</table>"
        )
