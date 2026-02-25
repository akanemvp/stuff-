"""
pitcher_card_app.py
────────────────────
Stuff+ Pitcher Card Dashboard

Tabs:
  1. 🏋 Train Model  — download 2025 Statcast data + calibrate formula
  2. 🔍 2026 Cards   — search any active pitcher, fetch live YTD data, score
  3. 📁 Upload Card  — score any manually-exported Savant CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, time
from pathlib import Path
from datetime import date, datetime

warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from savant_fetcher import (
    fetch_season, fetch_pitcher_ytd, fetch_pitcher_season,
    search_pitcher, cache_status
)
from model_trainer import (
    train, score, score_from_csv, model_is_trained, load_model,
    PHYSICS_VERSION, WEIGHTS
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stuff+  |  Pitch Quality Dashboard",
    page_icon="⚾",
    layout="wide",
)

CURRENT_YEAR = 2026
TRAIN_YEAR   = 2025

PITCH_NAMES = {
    "FF":"4-Seam","SI":"Sinker","FC":"Cutter","CH":"Changeup",
    "FS":"Splitter","FO":"Forkball","CU":"Curveball","KC":"Knuckle-Curve",
    "SL":"Slider","ST":"Sweeper","SW":"Sweeper","SV":"Slurve","KN":"Knuckleball","EP":"Eephus",
    "CS":"Slow Curve",
}

PITCH_COLOR = {
    "FF":"#E63946","SI":"#F4A261","FC":"#2A9D8F","CH":"#457B9D",
    "FS":"#1D3557","FO":"#264653","CU":"#6A4C93","KC":"#8338EC",
    "SL":"#FB8500","ST":"#FFB703","SW":"#FFB703","SV":"#E9C46A","KN":"#606C38","EP":"#DDA15E",
}

MODEL_DIR    = Path("data/model")
VERSION_FILE = MODEL_DIR / "physics_version.txt"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def stuff_color(v):
    if v >= 120: return "#1a7f37"
    if v >= 110: return "#2ea44f"
    if v >= 105: return "#4caf50"
    if v >= 100: return "#8bc34a"
    if v >=  95: return "#ffc107"
    if v >=  90: return "#ff9800"
    return "#f44336"

def grade_label(v):
    if v >= 130: return "Elite+"
    if v >= 120: return "Elite"
    if v >= 110: return "Plus+"
    if v >= 105: return "Plus"
    if v >= 100: return "Average"
    if v >=  95: return "Fringe"
    if v >=  90: return "Below Avg"
    return "Poor"

# ─── PITCH CARD RENDERER ──────────────────────────────────────────────────────

def render_card(df: pd.DataFrame, player_name: str, season: str,
                team: str, subtitle: str = ""):
    """Render a full pitcher card with movement plot and pitch table."""
    if "stuff_plus" not in df.columns or df["stuff_plus"].isna().all():
        st.warning("No Stuff+ scores found.")
        return

    df = df.copy()
    if "ivb" not in df.columns:
        df["ivb"] = df["pfx_z"] * 12 if "pfx_z" in df.columns else np.nan
    if "hb" not in df.columns:
        df["hb"]  = df["pfx_x"] * 12 if "pfx_x" in df.columns else np.nan

    # Per-pitch-type summary
    grp_cols = ["pitch_type","release_speed","ivb","hb","stuff_plus"]
    if "_vaa" in df.columns:       grp_cols.append("_vaa")
    if "adj_vaa" in df.columns:    grp_cols.append("adj_vaa")
    if "release_extension" in df.columns: grp_cols.append("release_extension")
    if "release_spin_rate" in df.columns: grp_cols.append("release_spin_rate")

    summary = (df[grp_cols]
               .groupby("pitch_type")
               .agg(
                   count    = ("stuff_plus","count"),
                   stuff_p  = ("stuff_plus","mean"),
                   velo     = ("release_speed","mean"),
                   ivb      = ("ivb","mean"),
                   hb       = ("hb","mean"),
                   **{c: (c,"mean") for c in ["_vaa","adj_vaa","release_extension",
                                               "release_spin_rate"] if c in grp_cols}
               )
               .reset_index()
               .sort_values("stuff_p", ascending=False))

    overall = df["stuff_plus"].mean()

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3,1,1])
    with c1:
        st.markdown(f"### ⚾ {player_name} — {season} {subtitle}")
        st.markdown(f"**Team:** {team}")
    with c2:
        col_val = stuff_color(overall)
        st.markdown(f"""
        <div style="text-align:center;background:{col_val};border-radius:12px;padding:12px">
            <div style="font-size:2rem;font-weight:700;color:white">{overall:.1f}</div>
            <div style="color:white;font-size:.85rem">Overall Stuff+</div>
            <div style="color:white;font-size:.75rem">{grade_label(overall)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="text-align:center;padding:12px;border:1px solid #ddd;border-radius:12px">
            <div style="font-size:1.4rem;font-weight:600">{len(summary)}</div>
            <div style="font-size:.8rem;color:#666">Pitch Types</div>
            <div style="font-size:1rem;font-weight:600">{int(df['stuff_plus'].count()):,}</div>
            <div style="font-size:.8rem;color:#666">Pitches</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Movement Plot ─────────────────────────────────────────────────────────
    col_plot, col_table = st.columns([1, 1])

    with col_plot:
        fig = go.Figure()

        # Background circles
        for r in [5, 10, 15, 20]:
            theta = np.linspace(0, 2*np.pi, 120)
            fig.add_trace(go.Scatter(
                x=r*np.cos(theta), y=r*np.sin(theta),
                mode="lines", line=dict(color="rgba(200,200,200,0.4)", width=1),
                showlegend=False, hoverinfo="skip"
            ))
        fig.add_hline(y=0, line=dict(color="rgba(150,150,150,0.4)", width=1))
        fig.add_vline(x=0, line=dict(color="rgba(150,150,150,0.4)", width=1))

        for _, row in summary.iterrows():
            pt   = row["pitch_type"]
            clr  = PITCH_COLOR.get(pt, "#888")
            sp   = row["stuff_p"]
            name = PITCH_NAMES.get(pt, pt)
            # Scatter individual pitches
            pt_mask = df["pitch_type"] == pt
            pt_df   = df[pt_mask]
            if len(pt_df) > 0:
                fig.add_trace(go.Scatter(
                    x=pt_df["hb"], y=pt_df["ivb"],
                    mode="markers",
                    marker=dict(color=clr, size=4, opacity=0.25),
                    name=name, legendgroup=pt, showlegend=False, hoverinfo="skip"
                ))
            # Mean marker sized by stuff+
            marker_size = 12 + max(0, (sp - 95)) * 0.8
            fig.add_trace(go.Scatter(
                x=[row["hb"]], y=[row["ivb"]],
                mode="markers+text",
                marker=dict(color=clr, size=marker_size,
                            line=dict(color="white", width=2)),
                text=[f"{name}<br>{sp:.0f}"],
                textposition="top center",
                textfont=dict(size=9, color=clr),
                name=name, legendgroup=pt, showlegend=True,
                hovertemplate=(f"<b>{name}</b><br>HB: %{{x:.1f}}\"<br>"
                               f"iVB: %{{y:.1f}}\"<br>Stuff+: {sp:.1f}<extra></extra>"),
            ))

        fig.update_layout(
            title="Movement Profile (pitcher's POV)",
            xaxis=dict(title="Horizontal Break (in)", range=[-25,25],
                       showgrid=False, zeroline=False),
            yaxis=dict(title="Induced Vertical Break (in)", range=[-25,25],
                       showgrid=False, zeroline=False, scaleanchor="x"),
            height=460, legend=dict(orientation="h", y=-0.12),
            margin=dict(l=40,r=20,t=50,b=60),
            plot_bgcolor="#fafafa", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pitch Table ───────────────────────────────────────────────────────────
    with col_table:
        st.markdown("#### Pitch Grades")
        for _, row in summary.iterrows():
            pt   = row["pitch_type"]
            sp   = row["stuff_p"]
            clr  = stuff_color(sp)
            name = PITCH_NAMES.get(pt, pt)
            pct  = df[df["pitch_type"]==pt].shape[0] / len(df) * 100
            velo = row.get("velo", np.nan)
            ivb  = row.get("ivb",  np.nan)
            hb   = row.get("hb",   np.nan)
            spin = row.get("release_spin_rate", np.nan) if "release_spin_rate" in row else np.nan

            velo_s = f"{velo:.1f}" if np.isfinite(velo) else "—"
            ivb_s  = f"{ivb:+.1f}" if np.isfinite(ivb)  else "—"
            hb_s   = f"{hb:+.1f}"  if np.isfinite(hb)   else "—"
            spin_s = f"{spin:.0f}" if np.isfinite(spin)  else "—"

            bar_w  = int(max(5, min(100, (sp - 70) / 60 * 100)))

            st.markdown(f"""
            <div style="margin:6px 0;padding:10px 14px;border-radius:10px;
                        background:linear-gradient(90deg,{clr}22 {bar_w}%,#f5f5f5 {bar_w}%);
                        border-left:4px solid {clr}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-weight:700;font-size:1rem">{name}</span>
                  <span style="color:#888;font-size:.8rem;margin-left:8px">{pct:.0f}%</span>
                </div>
                <div style="font-weight:700;font-size:1.3rem;color:{clr}">{sp:.1f}</div>
              </div>
              <div style="font-size:.78rem;color:#555;margin-top:3px">
                {velo_s} mph · iVB {ivb_s}" · HB {hb_s}" · {spin_s} rpm
              </div>
              <div style="font-size:.75rem;color:{clr};font-weight:600">{grade_label(sp)}</div>
            </div>""", unsafe_allow_html=True)

    # ── Stuff+ Bar Chart ──────────────────────────────────────────────────────
    st.markdown("#### Stuff+ by Pitch Type")
    fig2 = go.Figure()
    for _, row in summary.iterrows():
        pt   = row["pitch_type"]
        sp   = row["stuff_p"]
        name = PITCH_NAMES.get(pt, pt)
        clr  = PITCH_COLOR.get(pt, "#888")
        fig2.add_trace(go.Bar(
            x=[name], y=[sp], name=name,
            marker_color=clr,
            text=[f"{sp:.1f}"], textposition="outside",
        ))
    fig2.add_hline(y=100, line=dict(color="#888", dash="dash", width=1))
    fig2.update_layout(
        showlegend=False, height=280,
        yaxis=dict(title="Stuff+", range=[70, max(150, summary["stuff_p"].max()+10)]),
        margin=dict(l=40,r=20,t=20,b=40), plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Pitch Stats Table (matching screenshot format) ─────────────────────
    st.markdown("#### Pitch Arsenal Stats")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _clock(deg):
        try:
            d = float(deg) % 360
            h = int(d / 30) % 12 or 12
            m = int(((d / 30) % 1) * 60)
            return f"{h}:{m:02d}"
        except Exception:
            return "—"

    def _f(val, dec=1, comma=False):
        try:
            v = float(val)
            if np.isnan(v): return "—"
            return f"{int(round(v)):,}" if comma else f"{v:.{dec}f}"
        except Exception:
            return "—"

    def _badge(text, bg, fg="white", pad="2px 8px", fs=".78rem"):
        return (f'<span style="background:{bg};color:{fg};padding:{pad};'
                f'border-radius:5px;font-weight:700;font-size:{fs};'
                f'display:inline-block;line-height:1.5">{text}</span>')

    # ── aggregate ─────────────────────────────────────────────────────────────
    COL_MAP = [
        ("spin",    "release_spin_rate"),
        ("spineff", "spin_efficiency"),
        ("vaa",     "adj_vaa"),
        ("haa",     "_haa"),
        ("vrel",    "release_pos_z"),
        ("hrel",    "release_pos_x"),
        ("ext",     "release_extension"),
        ("spin_ax", "spin_axis_num"),
        ("ivb",     "ivb"),
        ("hb",      "hb"),
    ]
    agg_spec = {
        "count": ("stuff_plus", "count"),
        "sp":    ("stuff_plus", "mean"),
        "velo":  ("release_speed", "mean"),
    }
    for key, col in COL_MAP:
        if col in df.columns:
            agg_spec[key] = (col, "mean")

    tbl   = (df.groupby("pitch_type").agg(**agg_spec).reset_index()
               .sort_values("sp", ascending=False))
    total = int(tbl["count"].sum())

    # ── build HTML ────────────────────────────────────────────────────────────
    def cell(v, align="center", bold=False, color="#1a1a1a"):
        fw = "600" if bold else "400"
        return f'<td style="padding:7px 10px;text-align:{align};white-space:nowrap;font-weight:{fw};color:{color}">{v}</td>'

    COLS = ["Pitch","Count","Pitch%","Velo","iVB","HB",
            "Spin","Eff%","VAA","HAA","vRel","hRel","Ext","Axis","Stuff+","Grade"]

    hdr_cells = "".join(
        f'<th style="padding:8px 10px;text-align:center;font-size:.72rem;font-weight:700;'
        f'color:#555;border-bottom:2px solid #d0d4e0;white-space:nowrap;'
        f'background:#f0f2f7;letter-spacing:.3px">{c}</th>'
        for c in COLS
    )

    rows_html = []
    for i, row in tbl.reset_index(drop=True).iterrows():
        pt   = str(row["pitch_type"])
        name = PITCH_NAMES.get(pt, pt)
        clr  = PITCH_COLOR.get(pt, "#777")
        sp   = float(row["sp"])
        sp_c = stuff_color(sp)
        velo = float(row.get("velo", np.nan))
        cnt  = int(row["count"])
        pct  = cnt / total * 100
        bg   = "#f8f9fc" if i % 2 == 0 else "#ffffff"

        vc = ("#1a7f37" if velo >= 97 else "#2ea44f" if velo >= 94 else
              "#5b9e3e" if velo >= 91 else "#d97706" if velo >= 88 else "#dc2626")

        eff_v = float(row.get("spineff", np.nan)) if "spineff" in row.index else np.nan
        if np.isfinite(eff_v):
            ep   = eff_v * 100
            ec   = "#1565c0" if ep < 40 else "#555"
            eff_s = f'<b style="color:{ec}">{ep:.0f}%</b>' if ep < 40 else f'{ep:.0f}%'
        else:
            eff_s = "—"

        row_style = f'background:{bg}'
        r = (
            f'<tr style="{row_style}">'
            + cell(_badge(name, clr))
            + cell(f'{cnt:,}')
            + cell(f'{pct:.1f}%')
            + cell(_badge(f'{velo:.1f}', vc))
            + cell(_f(row.get("ivb",  np.nan) if "ivb"     in row.index else np.nan))
            + cell(_f(row.get("hb",   np.nan) if "hb"      in row.index else np.nan))
            + cell(_f(row.get("spin", np.nan) if "spin"    in row.index else np.nan, comma=True))
            + cell(eff_s)
            + cell(_f(row.get("vaa",  np.nan) if "vaa"     in row.index else np.nan))
            + cell(_f(row.get("haa",  np.nan) if "haa"     in row.index else np.nan))
            + cell(_f(row.get("vrel", np.nan) if "vrel"    in row.index else np.nan))
            + cell(_f(row.get("hrel", np.nan) if "hrel"    in row.index else np.nan))
            + cell(_f(row.get("ext",  np.nan) if "ext"     in row.index else np.nan))
            + cell(_clock(row.get("spin_ax", np.nan) if "spin_ax" in row.index else np.nan))
            + cell(_badge(f'{sp:.0f}', sp_c, pad="3px 9px", fs=".84rem"))
            + cell(f'<b style="color:{sp_c}">{grade_label(sp)}</b>')
            + '</tr>'
        )
        rows_html.append(r)

    table_html = (
        '<div style="overflow-x:auto;border-radius:10px;'
        'box-shadow:0 1px 5px rgba(0,0,0,.10);border:1px solid #d8dbe8;margin-top:4px">'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'font-family:ui-monospace,monospace;font-size:.79rem;color:#1a1a1a">'
        f'<thead><tr>{hdr_cells}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ─── SEASON LOGIC ─────────────────────────────────────────────────────────────
# 2026 season starts ~March 26. Before that, "Live" tab shows 2025 from cache.
LIVE_YEAR        = CURRENT_YEAR
SEASON_2026_START = date(2026, 3, 26)
SEASON_STARTED   = date.today() >= SEASON_2026_START
LIVE_TAB_LABEL   = f"🔍 {LIVE_YEAR} Cards" if SEASON_STARTED else "🔍 2026 Cards (pre-season)"

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_train, tab_2025, tab_live, tab_upload = st.tabs(
    ["🏋 Train Model", "📅 2025 Cards", LIVE_TAB_LABEL, "📁 Upload Card"]
)

# ── TAB 1: TRAIN ──────────────────────────────────────────────────────────────
with tab_train:
    st.header("Train Stuff+ Model")
    st.markdown("""
    Downloads **2025 MLB Statcast** data and calibrates the physics-formula scorer
    (league means/stds per pitch type). Takes ~2 min.
    """)

    # Stale model check
    if VERSION_FILE.exists():
        v = VERSION_FILE.read_text().strip()
        if v != PHYSICS_VERSION:
            st.warning(f"⚠️ Stale model detected (v: `{v}`). "
                       f"Current: `{PHYSICS_VERSION}`. Please retrain.")

    col1, col2 = st.columns([2,1])
    with col1:
        if st.button("🏋 Train / Retrain Model", type="primary", use_container_width=True):
            status = st.empty()
            prog   = st.progress(0)

            def status_fn(msg, pct=0.0):
                status.info(f"⏳ {msg}")
                prog.progress(min(int(pct*100), 100))

            try:
                status_fn("Fetching 2025 Statcast data…", 0.05)
                df_season = fetch_season(TRAIN_YEAR, status_fn=status_fn)
                summary   = train(df_season, status_fn=status_fn)
                prog.progress(100)
                status.success("✅ Model trained successfully!")
                st.markdown(f"""
                **Training Summary**
                - Pitches: `{summary['n_pitches']:,}`
                - Model: `{summary.get('model_type','Direct Formula')}`
                - Score p99: `{summary.get('score_p99','—')}` · p50: `{summary.get('score_p50','—')}` · p01: `{summary.get('score_p01','—')}`
                """)
                st.balloons()
            except Exception as e:
                status.error(f"Error: {e}")
                st.exception(e)

    with col2:
        cs = cache_status()
        st.markdown("**Cache Status**")
        for k, v in cs.items():
            st.markdown(f"- {k}: `{v}`")

    if model_is_trained():
        st.success(f"✅ Model ready — version `{PHYSICS_VERSION}`")
        st.markdown("**Pitch type weights (domain knowledge):**")
        rows = []
        for pt, w in WEIGHTS.items():
            rows.append({"Pitch": PITCH_NAMES.get(pt,pt), **{k:f"{v:.1f}" for k,v in w.items()}})
        st.dataframe(pd.DataFrame(rows).set_index("Pitch"), use_container_width=True)


# ── TAB 2: 2025 CARDS ────────────────────────────────────────────────────────
with tab_2025:
    st.header("2025 Pitcher Cards")
    st.markdown("Search any pitcher from the **2025 season** using the cached Statcast data.")

    if not model_is_trained():
        st.warning("⚠️ Model not trained yet. Go to **Train Model** tab first.")
    else:
        query_25 = st.text_input("Search pitcher name", placeholder="e.g. Corbin Burnes",
                                  key="search_2025")
        if not query_25:
            st.info("Enter a pitcher name to generate their 2025 card.")
        else:
            with st.spinner("Searching…"):
                results_25 = search_pitcher(query_25, year=2025)

            if not results_25:
                st.warning(f"No pitchers found for '{query_25}'")
            else:
                opts_25 = {f"{r['name']} (ID: {r['id']})": r for r in results_25}
                sel_25  = st.selectbox("Select pitcher", list(opts_25.keys()),
                                        key="select_2025")
                chosen_25 = opts_25[sel_25]

                if st.button("📅 Generate 2025 Card", type="primary", key="btn_2025"):
                    with st.spinner("Loading 2025 data from cache…"):
                        df_25 = fetch_pitcher_season(chosen_25["id"], year=2025)

                    if df_25 is None or len(df_25) == 0:
                        st.warning(
                            f"No 2025 data found for {chosen_25['name']}. "
                            "Make sure the model has been trained (which downloads and "
                            "caches the full 2025 season)."
                        )
                    else:
                        with st.spinner("Scoring…"):
                            try:
                                rank_norm, vaa_c, lg_stats, arm_models, haa_stats = load_model()
                                df_scored_25 = score(
                                    df_25,
                                    rank_norm=rank_norm, vaa_coeffs=vaa_c,
                                    league_stats=lg_stats, arm_models=arm_models,
                                    haa_stats=haa_stats,
                                )
                            except Exception as e:
                                st.error(f"Scoring error: {e}")
                                st.exception(e)
                                st.stop()

                        team_25 = ""
                        for c in ["home_team", "pitcher_team", "team"]:
                            if c in df_scored_25.columns:
                                team_25 = str(df_scored_25[c].mode()[0])
                                break

                        render_card(
                            df_scored_25, chosen_25["name"], "2025", team_25,
                            f"· {len(df_scored_25):,} pitches · full season"
                        )


# ── TAB 3: 2026 LIVE CARDS ────────────────────────────────────────────────────
with tab_live:
    st.header("2026 Pitcher Cards" if SEASON_STARTED else "2026 Cards — Season Not Started Yet")

    if not SEASON_STARTED:
        st.info(
            f"⏳ The 2026 season starts around **{SEASON_2026_START.strftime('%B %d, %Y')}**. "
            "Once it begins, live YTD cards will appear here automatically. "
            "Use **2025 Cards** in the meantime."
        )
        st.stop()

    if not model_is_trained():
        st.warning("⚠️ Model not trained yet. Go to **Train Model** tab first.")
        st.stop()

    # Search
    query = st.text_input("Search pitcher name", placeholder="e.g. Mason Miller")
    if not query:
        st.info("Enter a pitcher name above.")
        st.stop()

    with st.spinner("Searching…"):
        results = search_pitcher(query)

    if not results:
        st.warning(f"No pitchers found for '{query}'")
        st.stop()

    # Pitcher selector
    options = {f"{r['name']} (ID: {r['id']})": r for r in results}
    selected_label = st.selectbox("Select pitcher", list(options.keys()))
    selected = options[selected_label]

    if st.button("🔍 Generate Card", type="primary"):
        with st.spinner("Fetching 2026 YTD pitches…"):
            df_raw = fetch_pitcher_ytd(selected["id"], CURRENT_YEAR)

        if df_raw is None or len(df_raw) == 0:
            st.warning(f"No 2026 pitches found yet for {selected['name']}.")
            st.stop()

        with st.spinner("Scoring…"):
            try:
                rank_norm, vaa_c, lg_stats, arm_models, haa_stats = load_model()
                df_scored = score(df_raw, rank_norm=rank_norm, vaa_coeffs=vaa_c,
                                  league_stats=lg_stats, arm_models=arm_models, haa_stats=haa_stats)
            except Exception as e:
                st.error(f"Scoring error: {e}")
                st.exception(e)
                st.stop()

        team = ""
        for c in ["home_team","pitcher_team","team"]:
            if c in df_scored.columns:
                team = str(df_scored[c].mode()[0])
                break

        render_card(df_scored, selected["name"], str(CURRENT_YEAR), team,
                    f"· {len(df_scored)} pitches")


# ── TAB 3: UPLOAD ─────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload Pitcher CSV")
    st.markdown("Upload a CSV exported from [Baseball Savant](https://baseballsavant.mlb.com/).")

    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is None:
        st.stop()

    df_raw = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df_raw):,} rows")

    if st.button("⚡ Score Pitches", type="primary"):
        with st.spinner("Scoring…"):
            try:
                if model_is_trained():
                    rank_norm, vaa_c, lg_stats, arm_models, haa_stats = load_model()
                    df_scored = score(
                        score_from_csv.__wrapped__(df_raw) if hasattr(score_from_csv,'__wrapped__')
                        else score_from_csv(df_raw),
                        rank_norm=rank_norm, vaa_coeffs=vaa_c, league_stats=lg_stats
                    )
                    model_note = f"calibrated on {TRAIN_YEAR} Statcast"
                else:
                    df_scored = score_from_csv(df_raw)
                    model_note = "self-normalized (train model for league calibration)"
            except Exception as e:
                st.error(f"Scoring error: {e}")
                st.exception(e)
                st.stop()

        name_col = next((c for c in ["player_name","pitcher_name","name"]
                         if c in df_scored.columns), None)
        player_name = df_scored[name_col].iloc[0] if name_col else uploaded.name.replace(".csv","")
        yr_col   = next((c for c in ["game_year","year","season"]
                         if c in df_scored.columns), None)
        season_s = str(int(df_scored[yr_col].mode()[0])) if yr_col else "—"
        team_col = next((c for c in ["home_team","pitcher_team"] if c in df_scored.columns), None)
        team_s   = df_scored[team_col].mode()[0] if team_col else "—"

        st.markdown("---")
        render_card(df_scored, player_name, season_s, team_s, f"· {model_note}")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:48px;padding:16px;
            color:#2d3a50;font-family:'IBM Plex Mono';font-size:.7rem">
    Stuff+ · Physics-Based Pitch Quality · Data: Baseball Savant / Statcast<br>
    VAA computed via Statcast kinematics formula at y=1.417 ft (front of plate)<br>
    Calibrated on 2025 MLB regular-season pitches · μ=100, σ=10 (rank-based)
</div>
""", unsafe_allow_html=True)
