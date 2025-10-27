# streamlit run war-nlp-dashboard-app.py
# ──────────────────────────────────────────────────────────────────────────────
# WAR-NLP: “Is Science at War?” — Presentation Dashboard (Streamlit)
# Author: Sovesh
#
# Drop your CSVs into one folder per corpus:
#   results-pubmed/    → PubMed outputs
#   results-openalex/  → OpenAlex outputs
# Then run:
#   streamlit run war-nlp-dashboard-app.py
#
# The app is resilient to missing files (sections hide if data absent).
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import streamlit as st

# Altair is bundled with Streamlit
try:
    import altair as alt
    _HAS_ALTAIR = True
except Exception:
    _HAS_ALTAIR = False

# Avoid row caps; theme later
if _HAS_ALTAIR:
    try:
        alt.data_transformers.disable_max_rows()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Page config & style
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Is Science at War?", layout="wide", page_icon="")

ACCENT = "#6366F1"  # Indigo 500

st.markdown(
    f"""
    <style>
      :root {{ --accent: {ACCENT}; }}
      /* Accent only in main content (not in sidebar) */
      section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {{
        color: inherit !important;
      }}
      div.block-container h1, div.block-container h2 {{
        color: var(--accent) !important;
      }}
      .small {{font-size: 0.88rem; color: #555}}
      .mono {{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}}
      .tight {{margin-top: -0.4rem}}
      /* (Optional) If any experimental "talk" button appears in the sidebar, hide it. */
      section[data-testid="stSidebar"] [data-testid="stBaseButton-secondaryFormSubmit"] {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Altair theme using accent
if _HAS_ALTAIR:
    def _war_theme():
        return {
            "config": {
                "view": {"strokeWidth": 0},
                "axis": {"gridColor": "#eee", "labelColor": "#334155", "titleColor": "#111827"},
                "legend": {"labelColor": "#334155", "titleColor": "#111827"},
                "range": {
                    "category": [ACCENT, "#0ea5e9", "#10b981", "#ef4444", "#8b5cf6", "#f59e0b", "#22c55e", "#64748B"]
                },
            }
        }
    try:
        alt.themes.register("war_theme", _war_theme)
        alt.themes.enable("war_theme")
    except Exception:
        pass

# Distinct categorical palette (20)
DISTINCT_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd","#e6550d","#31a354","#756bb1","#636363",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — robust file loading & utilities
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path and path.exists():
            df = pd.read_csv(path)
            if "year" in df.columns and "pub_year" not in df.columns:
                df = df.rename(columns={"year": "pub_year"})
            return df
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
    return None

@st.cache_data(show_spinner=False)
def list_csvs(dir_path: Path) -> Dict[str, Path]:
    d: Dict[str, Path] = {}
    if not dir_path.exists():
        return d
    for p in dir_path.glob("*.csv"):
        d[p.name] = p
    return d

def x_year_axis(title: str = "Year"):
    if not _HAS_ALTAIR:
        return None
    try:
        yr0 = int(min(min_year, max_year))
        yr1 = int(max(min_year, max_year))
    except Exception:
        yr0, yr1 = 2010, 2025
    vals = list(range(yr0, yr1 + 1))
    return alt.X(
        "pub_year:Q",
        title=title,
        axis=alt.Axis(values=vals, format="d", labelOverlap=True, tickMinStep=1),
        scale=alt.Scale(domain=[yr0, yr1], nice=False),
    )

def line_with_band(df: pd.DataFrame, y: str, lo: Optional[str], hi: Optional[str], title: str, color: Optional[str] = None):
    if not _HAS_ALTAIR:
        st.line_chart(df.set_index("pub_year")[[y]])
        return
    base = alt.Chart(df)
    line = base.mark_line().encode(
        x=x_year_axis("Year"),
        y=alt.Y(f"{y}:Q", title=title),
        color=(alt.Color(color) if color else alt.value("steelblue")),
    )
    band = None
    if lo and hi and lo in df.columns and hi in df.columns:
        band = base.mark_area(opacity=0.18).encode(
            x=alt.X("pub_year:Q"), y=alt.Y(f"{lo}:Q"), y2=f"{hi}:Q",
            color=(alt.Color(color, legend=None) if color else alt.value("steelblue")),
        )
    st.altair_chart(((line + band) if band is not None else line).properties(height=340), use_container_width=True)

def _clip_years(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if "pub_year" in df.columns:
        try:
            df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
            df = df[df["pub_year"].between(min_year, max_year)].copy()
        except Exception:
            pass
    return df

def _drop_unknown(df: Optional[pd.DataFrame], cols: List[str]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c in df.columns:
            vals = df[c].astype(str).str.strip().str.lower()
            bad = vals.isin({"unknown", "nan", ""}) | df[c].isna()
            mask &= ~bad
    return df.loc[mask].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — single-corpus selector & filters (no directory pickers / no talk)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PUBMED_DIR = Path("results-pubmed")
DEFAULT_OPENALEX_DIR = Path("results-openalex")

st.sidebar.header("Data source")
source = st.sidebar.radio("Corpus", ["PubMed", "OpenAlex"], index=0, horizontal=True, key="src")

# Years hard-clamped to 2010–2025
min_year = int(st.sidebar.number_input("Min year", value=2010, step=1, min_value=2010, max_value=2025, key="ymin"))
max_year = int(st.sidebar.number_input("Max year", value=2025, step=1, min_value=2010, max_value=2025, key="ymax"))
if min_year > max_year:
    min_year, max_year = max_year, min_year

st.sidebar.header("Help")
SHOW_EXPLAINERS = st.sidebar.checkbox("Show overview text", value=True, help="Adds a short explainer at the top of every tab.")

# Fixed directories (hidden from sidebar)
base_dir = DEFAULT_PUBMED_DIR if source == "PubMed" else DEFAULT_OPENALEX_DIR
available = list_csvs(base_dir)

# ──────────────────────────────────────────────────────────────────────────────
# Expected files map
# ──────────────────────────────────────────────────────────────────────────────
FILES = {
    # Overview
    "std_any":   ["prevalence_year_core_standardized_from_clean_any.csv", "prevalence_year_core_standardized_from_clean.csv"],
    "std_title": ["prevalence_year_core_standardized_from_clean_title.csv"],
    "std_abs":   ["prevalence_year_core_standardized_from_clean_abstract.csv"],
    "raw_any":   ["prevalence_year_core_raw_from_clean_any.csv", "prevalence_year_core_raw_from_clean.csv"],
    "raw_title": ["prevalence_year_core_raw_from_clean_title.csv"],
    "raw_abs":   ["prevalence_year_core_raw_from_clean_abstract.csv"],

    # Title vs Abstract
    "title_vs_abs": ["prevalence_title_vs_abstract_core_from_clean.csv"],

    # Doc types
    "dt_title":       ["prevalence_year_title_core_by_doctype_from_clean.csv"],
    "dt_abs":         ["prevalence_year_abstract_core_by_doctype_from_clean.csv"],
    "dt_sizes_title": ["doctype_sizes_core_title_from_clean.csv"],
    "dt_sizes_abs":   ["doctype_sizes_core_abstract_from_clean.csv"],
    "dt_or_title":    ["doctype_trend_or_title_core_agg_from_clean.csv"],
    "dt_or_abs":      ["doctype_trend_or_abstract_core_agg_from_clean.csv"],
}

def resolve_file(key: str) -> Optional[Path]:
    for cand in FILES.get(key, []):
        p = base_dir / cand
        if p.exists():
            return p
    return None

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
st.title("Is Science at War?")
st.caption(f"Corpus: **{source}** · Data dir: `{base_dir}`")

T_OVERVIEW, T_TVA, T_DT = st.tabs(["Overview", "Title vs Abstract", "Doc Types"])

# ──────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with T_OVERVIEW:
    st.subheader("Global prevalence over time")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Share of papers each year containing any ‘war’ lexeme.\n\n"
            "**Two versions.** *Raw prevalence* is observed; *Length-standardized* estimates the share if every doc had L* words.\n\n"
            "**How to read.** Upward line ⇒ more frequent mentions. Shaded bands are 95% CIs (if available)."
        )

    view = st.radio("Slice", ["Any (union)", "Title", "Abstract"], horizontal=True, key="ov_slice")

    if view == "Any (union)":
        p_std = resolve_file("std_any");   p_raw = resolve_file("raw_any")
    elif view == "Title":
        p_std = resolve_file("std_title"); p_raw = resolve_file("raw_title")
    else:
        p_std = resolve_file("std_abs");   p_raw = resolve_file("raw_abs")

    df_std = _clip_years(load_csv_optional(p_std) if p_std else None)
    df_raw = _clip_years(load_csv_optional(p_raw) if p_raw else None)

    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    try:
        if df_raw is not None and {"pub_year","prevalence"}.issubset(df_raw.columns):
            y_max = int(pd.to_numeric(df_raw["pub_year"]).max())
            last = float(df_raw.loc[df_raw["pub_year"] == y_max, "prevalence"].mean())
            prev = float(df_raw.loc[df_raw["pub_year"] == y_max-1, "prevalence"].mean()) if (df_raw["pub_year"] == y_max-1).any() else float("nan")
            delta = None if pd.isna(prev) else last - prev
            m1.metric("Last year prevalence", f"{last:.2%}", None if delta is None else f"{delta:+.2%}")
        if df_raw is not None and "n_docs" in df_raw.columns:
            total_docs = int(pd.to_numeric(df_raw["n_docs"]).sum())
            m2.metric("Total docs (raw series)", f"{total_docs:,}")
        if df_std is not None and "L_star" in df_std.columns and not df_std["L_star"].isna().all():
            Ls = int(pd.to_numeric(df_std["L_star"]).dropna().iloc[0])
            m3.metric("Standardized at L*", f"{Ls} words")
        if df_raw is not None and "pub_year" in df_raw.columns:
            y_min = int(pd.to_numeric(df_raw["pub_year"]).min())
            y_max_kpi = int(pd.to_numeric(df_raw["pub_year"]).max())
            m4.metric("Years covered", f"{y_min}–{y_max_kpi}")
    except Exception:
        pass

    cols = st.columns(2)
    with cols[0]:
        if df_raw is not None and {"pub_year","prevalence"}.issubset(df_raw.columns):
            st.markdown("**Raw prevalence**")
            line_with_band(df_raw, y="prevalence", lo="prev_lo95", hi="prev_hi95", title="Prevalence")
        else:
            st.info("Raw prevalence CSV not found or missing columns.")

    with cols[1]:
        if df_std is not None and {"pub_year","std_prev"}.issubset(df_std.columns):
            st.markdown("**Length-standardized prevalence**")
            line_with_band(df_std, y="std_prev", lo="std_lo95", hi="std_hi95", title="Standardized prevalence")
            if "L_star" in df_std.columns:
                Ls = sorted(df_std["L_star"].dropna().unique())
                st.caption(f"Standardized at L* = {Ls[0] if Ls else '—'} words.")
        else:
            st.info("Standardized prevalence CSV not found or missing columns.")

# ──────────────────────────────────────────────────────────────────────────────
# TITLE vs ABSTRACT
# ──────────────────────────────────────────────────────────────────────────────
with T_TVA:
    st.subheader("Where do war-terms show up — titles or abstracts?")
    if SHOW_EXPLAINERS:
        st.info(
            "Direct comparison of title vs abstract prevalence. Divergence can indicate changes in how prominently the topic is framed."
        )

    p_tva = resolve_file("title_vs_abs")
    df_tva = _clip_years(load_csv_optional(p_tva) if p_tva else None)
    if df_tva is None:
        st.info("Title vs Abstract CSV not found.")
    else:
        if _HAS_ALTAIR and {"prev_title","prev_abstract"}.issubset(df_tva.columns):
            dfm = df_tva.melt("pub_year", value_vars=["prev_title","prev_abstract"], var_name="series", value_name="prevalence")
            ch = alt.Chart(dfm).mark_line().encode(
                x=x_year_axis("Year"),
                y=alt.Y("prevalence:Q", title="Prevalence"),
                color=alt.Color("series:N", title="", scale=alt.Scale(scheme="set1")),
                tooltip=["pub_year","series","prevalence"],
            ).properties(height=360)
            st.altair_chart(ch, use_container_width=True)
        else:
            st.dataframe(df_tva)

# ──────────────────────────────────────────────────────────────────────────────
# DOC TYPES (K-slider fix: dynamic defaults + robust fallbacks)
# ──────────────────────────────────────────────────────────────────────────────
with T_DT:
    st.subheader("Document types")
    if SHOW_EXPLAINERS:
        st.info(
            "Prevalence by publication type (e.g., Article, Review, Editorial). "
            "Use the **Top-K** slider to quickly select the most common types."
        )

    left, right = st.columns([1, 1])

    with left:
        which = st.radio("Slice", ["Title", "Abstract"], horizontal=True, key="dt_slice")
        p_sizes = resolve_file("dt_sizes_title" if which == "Title" else "dt_sizes_abs")
        p_yearly = resolve_file("dt_title" if which == "Title" else "dt_abs")
        p_or = resolve_file("dt_or_title" if which == "Title" else "dt_or_abs")

        df_sizes = load_csv_optional(p_sizes) if p_sizes else None
        df_year  = _clip_years(load_csv_optional(p_yearly) if p_yearly else None)

        # Normalize doc_type column names across files
        def _norm_doctype(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            df = df.copy()
            if "doc_type" not in df.columns and "doc_type_major" in df.columns:
                df = df.rename(columns={"doc_type_major": "doc_type"})
            return df
        df_sizes = _norm_doctype(_drop_unknown(df_sizes, ["doc_type"]))
        df_year  = _norm_doctype(_drop_unknown(df_year,  ["doc_type"]))

        # Fallback if sizes file missing or lacks n_docs:
        if (df_sizes is None) or ("n_docs" not in df_sizes.columns or df_sizes["n_docs"].isna().all()):
            if df_year is not None:
                if "n_docs" in df_year.columns and not df_year["n_docs"].isna().all():
                    df_sizes = (df_year.groupby("doc_type", as_index=False)["n_docs"].sum()
                                      .sort_values("n_docs", ascending=False))
                else:
                    # fallback to occurrence counts
                    df_sizes = (df_year["doc_type"].value_counts()
                                .rename_axis("doc_type").reset_index(name="n_docs"))
            # else: remains None

        if df_sizes is not None and "doc_type" in df_sizes.columns:
            df_sizes = df_sizes.sort_values("n_docs", ascending=False)
            max_k = int(min(12, max(3, len(df_sizes))))
            topk = st.slider("Show top K by total N", 3, max_k, min(6, max_k), key="dt_topk")

            # dynamic defaults tied to K (the key ensures reset when K changes)
            default_list = df_sizes["doc_type"].head(topk).tolist()
            sel = st.multiselect(
                "Doc types",
                df_sizes["doc_type"].head(20).tolist(),
                default=default_list,
                key=f"dt_sel_{source}_{which}_{topk}",
            )
        else:
            sel = []
            st.info("Doc-type sizes CSV not found; will plot whatever appears in yearly data if selected.")

    with right:
        if df_year is not None and sel:
            df_plot = df_year[df_year["doc_type"].isin(sel)].copy()

            if _HAS_ALTAIR and {"prevalence"}.issubset(df_plot.columns):
                base = alt.Chart(df_plot)
                color_dt = alt.Color("doc_type:N", title="Doc type", scale=alt.Scale(range=DISTINCT_PALETTE))
                line = base.mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("prevalence:Q", title="Prevalence"),
                    color=color_dt,
                )
                overlay = line
                if {"prev_lo95","prev_hi95"}.issubset(df_plot.columns):
                    band = base.mark_area(opacity=0.18).encode(
                        x=alt.X("pub_year:Q"),
                        y="prev_lo95:Q",
                        y2="prev_hi95:Q",
                        color=alt.Color("doc_type:N", scale=alt.Scale(range=DISTINCT_PALETTE), legend=None),
                    )
                    overlay = band + line
                st.altair_chart(overlay.properties(height=360), use_container_width=True)
            else:
                st.dataframe(df_plot)
        else:
            st.info("Select some doc types to plot annual prevalence.")

    st.markdown("**Aggregated-GLM trends (OR per year)**")
    df_or = load_csv_optional(p_or) if p_or else None
    df_or = _norm_doctype(_drop_unknown(df_or, ["doc_type"]))
    if df_or is not None:
        st.dataframe(df_or.sort_values(["p", "N_total"]).reset_index(drop=True))
    else:
        st.info("Doc-type ORs CSV not found.")

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="small">Copyright reserved by Complex Systems Lab @ Penn</div>', unsafe_allow_html=True)
