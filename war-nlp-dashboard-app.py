# streamlit run war-nlp-dashboard-app.py
# ──────────────────────────────────────────────────────────────────────────────
# WAR-NLP: “Is Science at War?” — Presentation Dashboard (Streamlit)
# Author: Sovesh
#
# Drop your CSVs into two folders:
#   results-pubmed/    → PubMed outputs
#   results-openalex/  → OpenAlex outputs
# Then run:
#   streamlit run war-nlp-dashboard-app.py
#
# This app is resilient to missing files. It will show only the sections
# that can be populated from whatever CSVs it finds.
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

# Avoid row caps; we’ll set a theme later
if _HAS_ALTAIR:
    try:
        alt.data_transformers.disable_max_rows()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Page config & small style tweaks
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Is Science at War?",
    layout="wide",
    page_icon="",
)

st.markdown(
    """
    <style>
      .small {font-size: 0.88rem; color: #555}
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
      .tight {margin-top: -0.4rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — robust file loading, caching, and small utilities
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

@st.cache_data(show_spinner=False)
def wilson_ci(k, n, z=1.959963984540054):
    import numpy as np
    k = k.astype(float)
    n = n.astype(float).clip(1)
    ph = k / n
    denom = 1 + z**2 / n
    ctr = (ph + z**2/(2*n)) / denom
    half= z*((ph*(1-ph) + z**2/(4*n))/n)**0.5 / denom
    return ctr-half, ctr+half

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

def line_with_band(
    df: pd.DataFrame,
    y: str,
    lo: Optional[str],
    hi: Optional[str],
    title: str,
    color: Optional[str] = None,
):
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
            x=alt.X("pub_year:Q"),
            y=alt.Y(f"{lo}:Q"),
            y2=f"{hi}:Q",
            color=(alt.Color(color, legend=None) if color else alt.value("steelblue")),
        )

    st.altair_chart(((line + band) if band is not None else line).properties(height=340),
                    use_container_width=True)

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
# Sidebar — choose source + quick filters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PUBMED_DIR = Path("results-pubmed")
DEFAULT_OPENALEX_DIR = Path("results-openalex")

st.sidebar.header("Data sources")
source = st.sidebar.radio(
    "Corpus",
    ["PubMed", "OpenAlex", "Both (compare)"],
    index=0,
    horizontal=True,
    key="src",
)

# Hard clamp year inputs to 2010–2025 (cannot go outside)
min_year = int(st.sidebar.number_input(
    "Min year", value=2010, step=1, min_value=2010, max_value=2025, key="ymin"
))
max_year = int(st.sidebar.number_input(
    "Max year", value=2025, step=1, min_value=2010, max_value=2025, key="ymax"
))
if min_year > max_year:
    min_year, max_year = max_year, min_year

# Help / explainer toggle
st.sidebar.header("Help")
SHOW_EXPLAINERS = st.sidebar.checkbox(
    "Show overview text",
    value=True,
    help="Adds a short explainer at the top of every tab."
)

# Accent fixed to Indigo
ACCENT = "#6366F1"

# High-contrast categorical palette for distinct series (20 colors)
DISTINCT_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd","#e6550d","#31a354","#756bb1","#636363",
]

# Inject accent CSS
st.markdown(
    f"""
    <style>
      :root {{ --accent: {ACCENT}; }}
      h1, h2 {{ color: var(--accent) !important; }}
      .stTabs [data-baseweb="tab"] p {{ font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Altair theme with accent
if _HAS_ALTAIR:
    def _war_theme():
        return {
            "config": {
                "view": {"strokeWidth": 0},
                "axis": {"gridColor": "#eee", "labelColor": "#334155", "titleColor": "#111827"},
                "legend": {"labelColor": "#334155", "titleColor": "#111827"},
                "range": {"category": [ACCENT, "#0ea5e9", "#10b981", "#ef4444", "#8b5cf6", "#f59e0b", "#22c55e", "#64748B"]},
            }
        }
    try:
        alt.themes.register("war_theme", _war_theme)
        alt.themes.enable("war_theme")
    except Exception:
        pass

# Corpus directories
CORPUS_DIRS = {
    "PubMed": DEFAULT_PUBMED_DIR,
    "OpenAlex": DEFAULT_OPENALEX_DIR,
}
is_compare = source.startswith("Both")
active_corpora = ["PubMed", "OpenAlex"] if is_compare else [source]

# Convenience: per-corpus file resolver/loaders
def resolve_file_for(key: str, corpus: str) -> Optional[Path]:
    FILES = {
        "std_any":   ["prevalence_year_core_standardized_from_clean_any.csv", "prevalence_year_core_standardized_from_clean.csv"],
        "std_title": ["prevalence_year_core_standardized_from_clean_title.csv"],
        "std_abs":   ["prevalence_year_core_standardized_from_clean_abstract.csv"],
        "raw_any":   ["prevalence_year_core_raw_from_clean_any.csv", "prevalence_year_core_raw_from_clean.csv"],
        "raw_title": ["prevalence_year_core_raw_from_clean_title.csv"],
        "raw_abs":   ["prevalence_year_core_raw_from_clean_abstract.csv"],

        "title_vs_abs": ["prevalence_title_vs_abstract_core_from_clean.csv"],

        "dt_title":       ["prevalence_year_title_core_by_doctype_from_clean.csv"],
        "dt_abs":         ["prevalence_year_abstract_core_by_doctype_from_clean.csv"],
        "dt_sizes_title": ["doctype_sizes_core_title_from_clean.csv"],
        "dt_sizes_abs":   ["doctype_sizes_core_abstract_from_clean.csv"],
        "dt_or_title":    ["doctype_trend_or_title_core_agg_from_clean.csv"],
        "dt_or_abs":      ["doctype_trend_or_abstract_core_agg_from_clean.csv"],

        "cty_any":     ["prevalence_year_core_by_country_from_clean_union.csv", "prevalence_year_core_by_country_from_clean.csv"],
        "cty_title":   ["prevalence_year_core_by_country_from_clean_title.csv"],
        "cty_abs":     ["prevalence_year_core_by_country_from_clean_abstract.csv"],
        "cty_or_any":  ["country_trend_or_core_agg_from_clean.csv"],
        "cty_or_title":["country_trend_or_core_agg_from_clean_title.csv"],
        "cty_or_abs":  ["country_trend_or_core_agg_from_clean_abstract.csv"],
        "cty_sizes":   ["country_sizes_core_from_clean.csv"],

        "lexemes": ["prevalence_year_core_lexemes_from_clean.csv"],

        "mshare_year": ["metaphor_share_by_year_union.csv"],
        "mshare_dt":   ["metaphor_share_by_doctype_union.csv"],
        "mshare_cty":  ["metaphor_share_by_country_union.csv"],

        "coef_any":   ["coef_core_glm_from_clean_any.csv", "coef_core_glm_from_clean.csv"],
        "coef_title": ["coef_core_glm_from_clean_title.csv"],
        "coef_abs":   ["coef_core_glm_from_clean_abstract.csv"],
    }
    base = CORPUS_DIRS[corpus]
    for cand in FILES.get(key, []):
        p = base / cand
        if p.exists():
            return p
    return None

def load_df_for(key: str, corpus: str) -> Optional[pd.DataFrame]:
    p = resolve_file_for(key, corpus)
    df = _clip_years(load_csv_optional(p) if p else None)
    if df is not None:
        df = df.copy()
        df["corpus"] = corpus
    return df

def load_df_all(key: str) -> Dict[str, Optional[pd.DataFrame]]:
    return {c: load_df_for(key, c) for c in active_corpora}

# Header / caption
if is_compare:
    st.title("Is Science at War?")
    st.caption(f"Corpora: **PubMed** (`{CORPUS_DIRS['PubMed']}`) · **OpenAlex** (`{CORPUS_DIRS['OpenAlex']}`)")
else:
    st.title("Is Science at War?")
    st.caption(f"Corpus: **{active_corpora[0]}** · Data dir: `{CORPUS_DIRS[active_corpora[0]]}`")

T_OVERVIEW, T_TVA, T_DT, T_CTY, T_LEX, T_MS, T_GLM, T_DL = st.tabs(
    ["Overview", "Title vs Abstract", "Doc Types", "Countries", "Lexemes", "Contextual", "GLM Coefs", "Downloads"]
)

# ──────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with T_OVERVIEW:
    st.subheader("Global prevalence over time")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** The share of papers each year that contain any term "
            "from the core ‘war’ lexeme list.\n\n"
            "**Two versions.** *Raw prevalence* is the observed share in your data. "
            "*Length-standardized prevalence* estimates the share if every document had L* words, "
            "so trends aren’t driven by changing abstract lengths.\n\n"
            "**Compare.** Choose *Both (compare)* above to overlay PubMed and OpenAlex."
        )

    view = st.radio("Slice", ["Any (union)", "Title", "Abstract"], horizontal=True, key="ov_slice")

    key_std = "std_any" if view == "Any (union)" else ("std_title" if view == "Title" else "std_abs")
    key_raw = "raw_any" if view == "Any (union)" else ("raw_title" if view == "Title" else "raw_abs")

    if not is_compare:
        df_std = load_df_for(key_std, active_corpora[0])
        df_raw = load_df_for(key_raw, active_corpora[0])

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
                y_max = int(pd.to_numeric(df_raw["pub_year"]).max())
                m4.metric("Years covered", f"{y_min}–{y_max}")
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
    else:
        dfs_raw = [d for d in load_df_all(key_raw).values() if d is not None]
        dfs_std = [d for d in load_df_all(key_std).values() if d is not None]

        cols = st.columns(2)

        if _HAS_ALTAIR:
            with cols[0]:
                if dfs_raw:
                    df_all = pd.concat(dfs_raw, ignore_index=True)
                    st.markdown("**Raw prevalence — PubMed vs OpenAlex**")
                    base = alt.Chart(df_all)
                    band = None
                    if {"prev_lo95","prev_hi95"}.issubset(df_all.columns):
                        band = base.mark_area(opacity=0.15).encode(
                            x=alt.X("pub_year:Q"),
                            y="prev_lo95:Q",
                            y2="prev_hi95:Q",
                            color=alt.Color("corpus:N", legend=None),
                            detail="corpus:N",
                        )
                    line = base.mark_line().encode(
                        x=x_year_axis("Year"),
                        y=alt.Y("prevalence:Q", title="Prevalence"),
                        color=alt.Color("corpus:N", title="Corpus"),
                        strokeDash=alt.StrokeDash("corpus:N", legend=None),
                        tooltip=["pub_year","corpus","prevalence"],
                    )
                    st.altair_chart(((band + line) if band is not None else line).properties(height=340),
                                    use_container_width=True)
                else:
                    st.info("Raw prevalence CSVs not found for either corpus.")

            with cols[1]:
                if dfs_std:
                    df_all = pd.concat(dfs_std, ignore_index=True)
                    st.markdown("**Length-standardized prevalence — PubMed vs OpenAlex**")
                    base = alt.Chart(df_all)
                    band = None
                    if {"std_lo95","std_hi95"}.issubset(df_all.columns):
                        band = base.mark_area(opacity=0.15).encode(
                            x=alt.X("pub_year:Q"),
                            y="std_lo95:Q",
                            y2="std_hi95:Q",
                            color=alt.Color("corpus:N", legend=None),
                            detail="corpus:N",
                        )
                    line = base.mark_line().encode(
                        x=x_year_axis("Year"),
                        y=alt.Y("std_prev:Q", title="Standardized prevalence"),
                        color=alt.Color("corpus:N", title="Corpus"),
                        strokeDash=alt.StrokeDash("corpus:N", legend=None),
                        tooltip=["pub_year","corpus","std_prev"],
                    )
                    st.altair_chart(((band + line) if band is not None else line).properties(height=340),
                                    use_container_width=True)
                else:
                    st.info("Standardized prevalence CSVs not found for either corpus.")
        else:
            st.info("Altair unavailable; comparison plotting disabled.")

# ──────────────────────────────────────────────────────────────────────────────
# TITLE vs ABSTRACT
# ──────────────────────────────────────────────────────────────────────────────
with T_TVA:
    st.subheader("Where do war-terms show up — titles or abstracts?")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Title vs abstract prevalence. "
            "Use *Both (compare)* to overlay corpora (color by series, dashed by corpus)."
        )

    if not is_compare:
        df_tva = load_df_for("title_vs_abs", active_corpora[0])
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
    else:
        dfs = [d for d in load_df_all("title_vs_abs").values() if d is not None]
        if not dfs:
            st.info("Title vs Abstract CSVs not found.")
        elif _HAS_ALTAIR:
            df_all = pd.concat(dfs, ignore_index=True)
            dfm = df_all.melt(["pub_year","corpus"], value_vars=["prev_title","prev_abstract"],
                              var_name="series", value_name="prevalence")
            ch = alt.Chart(dfm).mark_line().encode(
                x=x_year_axis("Year"),
                y=alt.Y("prevalence:Q", title="Prevalence"),
                color=alt.Color("series:N", title="Series"),
                strokeDash=alt.StrokeDash("corpus:N", title="Corpus"),
                tooltip=["pub_year","series","corpus","prevalence"],
            ).properties(height=360)
            st.altair_chart(ch, use_container_width=True)
        else:
            st.dataframe(pd.concat(dfs, ignore_index=True))

# ──────────────────────────────────────────────────────────────────────────────
# DOC TYPES
# ──────────────────────────────────────────────────────────────────────────────
with T_DT:
    st.subheader("Document types")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Prevalence by publication type. "
            "In compare mode, lines are colored by doc type and **dashed by corpus**."
        )

    left, right = st.columns([1, 1])
    with left:
        which = st.radio("Slice", ["Title", "Abstract"], horizontal=True, key="dt_slice")

        sizes_key = "dt_sizes_title" if which == "Title" else "dt_sizes_abs"
        sizes_dfs = [d for d in load_df_all(sizes_key).values() if d is not None]
        if sizes_dfs:
            df_sizes_all = pd.concat(sizes_dfs, ignore_index=True)
            if "doc_type" not in df_sizes_all.columns and "doc_type_major" in df_sizes_all.columns:
                df_sizes_all = df_sizes_all.rename(columns={"doc_type_major": "doc_type"})
            if "n_docs" in df_sizes_all.columns:
                top_counts = (df_sizes_all.groupby("doc_type")["n_docs"].sum()
                              .sort_values(ascending=False))
                topk_default = top_counts.head(6).index.tolist()
                full_choices = top_counts.head(20).index.tolist()
            else:
                full_choices = sorted(df_sizes_all["doc_type"].dropna().unique().tolist())[:20]
                topk_default = full_choices[:6]
        else:
            df_sizes_all = None
            full_choices, topk_default = [], []

        topk = st.slider("Show top K by total N", 3, 12, min(6, max(3, len(topk_default))) if topk_default else 6, key="dt_topk")
        default_list = (topk_default[:topk] if topk_default else [])
        sel = st.multiselect(
            "Doc types",
            full_choices,
            default=default_list,
            key=f"dt_sel_{which}_{topk}_{'both' if is_compare else active_corpora[0]}",
        )

    with right:
        yearly_key = "dt_title" if which == "Title" else "dt_abs"
        if not is_compare:
            df_year = load_df_for(yearly_key, active_corpora[0])
            df_year = _drop_unknown(df_year, ["doc_type_major","doc_type"])
            if df_year is not None and sel:
                df_plot = df_year[df_year.get("doc_type_major", df_year.get("doc_type")).isin(sel)].copy()
                if "doc_type" not in df_plot.columns and "doc_type_major" in df_plot.columns:
                    df_plot = df_plot.rename(columns={"doc_type_major": "doc_type"})
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
                st.info("Select some doc types to plot.")
        else:
            dfs_year = [d for d in load_df_all(yearly_key).values() if d is not None]
            if dfs_year and sel and _HAS_ALTAIR:
                df_plot = pd.concat(dfs_year, ignore_index=True)
                if "doc_type" not in df_plot.columns and "doc_type_major" in df_plot.columns:
                    df_plot = df_plot.rename(columns={"doc_type_major": "doc_type"})
                df_plot = df_plot[df_plot["doc_type"].isin(sel)]
                base = alt.Chart(df_plot)
                ch = base.mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("prevalence:Q", title="Prevalence"),
                    color=alt.Color("doc_type:N", title="Doc type", scale=alt.Scale(range=DISTINCT_PALETTE)),
                    strokeDash=alt.StrokeDash("corpus:N", title="Corpus"),
                    tooltip=["pub_year","doc_type","corpus","prevalence"],
                ).properties(height=360)
                st.altair_chart(ch, use_container_width=True)
            elif not sel:
                st.info("Select some doc types to plot.")
            else:
                st.info("Doc-type CSVs not found for either corpus.")

    st.markdown("**Aggregated-GLM trends (OR per year)**")
    or_key = "dt_or_title" if which == "Title" else "dt_or_abs"
    if not is_compare:
        df_or = _drop_unknown(load_df_for(or_key, active_corpora[0]), ["doc_type","doc_type_major"])
        if df_or is not None:
            if "doc_type_major" in df_or.columns and "doc_type" not in df_or.columns:
                df_or = df_or.rename(columns={"doc_type_major": "doc_type"})
            st.dataframe(df_or.sort_values(["p", "N_total"]).reset_index(drop=True))
        else:
            st.info("Doc-type ORs CSV not found.")
    else:
        dfs_or = [d for d in load_df_all(or_key).values() if d is not None]
        if dfs_or:
            df_or = pd.concat(dfs_or, ignore_index=True)
            if "doc_type_major" in df_or.columns and "doc_type" not in df_or.columns:
                df_or = df_or.rename(columns={"doc_type_major": "doc_type"})
            st.dataframe(df_or.sort_values(["corpus","p","N_total"]).reset_index(drop=True))
        else:
            st.info("Doc-type ORs CSVs not found.")

# ──────────────────────────────────────────────────────────────────────────────
# COUNTRIES
# ──────────────────────────────────────────────────────────────────────────────
with T_CTY:
    st.subheader("Countries")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Prevalence by country. "
            "In compare mode, lines are colored by country and **dashed by corpus**."
        )

    c_slice = st.radio("Slice", ["Any (union)", "Title", "Abstract"], horizontal=True, key="cty_slice")

    key_cty = {"Any (union)": "cty_any", "Title": "cty_title", "Abstract": "cty_abs"}[c_slice]
    key_or = {"Any (union)": "cty_or_any", "Title": "cty_or_title", "Abstract": "cty_or_abs"}[c_slice]
    key_sizes = "cty_sizes"

    sizes_dfs = [d for d in load_df_all(key_sizes).values() if d is not None]
    if sizes_dfs:
        df_sizes_all = pd.concat(sizes_dfs, ignore_index=True)
        if "country" not in df_sizes_all.columns and "country_primary" in df_sizes_all.columns:
            df_sizes_all = df_sizes_all.rename(columns={"country_primary": "country"})
        if "n_docs" in df_sizes_all.columns:
            top_c = (df_sizes_all.groupby("country")["n_docs"].sum()
                     .sort_values(ascending=False))
            default_c = top_c.head(8).index.tolist()
            choices = top_c.index.tolist()
        else:
            choices = sorted(df_sizes_all["country"].dropna().unique().tolist())
            default_c = choices[:8]
    else:
        choices, default_c = [], []

    sel_c = st.multiselect("Countries", choices, default=default_c, key="cty_sel")

    if not is_compare:
        df_cty = _drop_unknown(load_df_for(key_cty, active_corpora[0]), ["country","country_primary"])
        if df_cty is not None and sel_c:
            dfp = df_cty[df_cty.get("country_primary", df_cty.get("country")).isin(sel_c)].copy()
            if "country" not in dfp.columns and "country_primary" in dfp.columns:
                dfp = dfp.rename(columns={"country_primary": "country"})
            if _HAS_ALTAIR and {"prevalence"}.issubset(dfp.columns):
                base = alt.Chart(dfp)
                line = base.mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("prevalence:Q", title="Prevalence"),
                    color=alt.Color("country:N", title="Country", scale=alt.Scale(range=DISTINCT_PALETTE)),
                )
                overlay = line
                if {"prev_lo95","prev_hi95"}.issubset(dfp.columns):
                    band = base.mark_area(opacity=0.18).encode(
                        x=alt.X("pub_year:Q"),
                        y="prev_lo95:Q",
                        y2="prev_hi95:Q",
                        color=alt.Color("country:N", scale=alt.Scale(range=DISTINCT_PALETTE), legend=None),
                    )
                    overlay = band + line
                st.altair_chart(overlay.properties(height=360), use_container_width=True)
            else:
                st.dataframe(dfp)
        else:
            st.info("Select some countries to plot annual prevalence.")

        df_or = _drop_unknown(load_df_for(key_or, active_corpora[0]), ["country"])
        st.markdown("**Aggregated-GLM trends (OR per year)**")
        if df_or is not None:
            st.dataframe(df_or.sort_values(["p", "N_total"]).reset_index(drop=True))
        else:
            st.info("Country ORs CSV not found.")
    else:
        dfs_cty = [d for d in load_df_all(key_cty).values() if d is not None]
        if dfs_cty and sel_c and _HAS_ALTAIR:
            dfp = pd.concat(dfs_cty, ignore_index=True)
            if "country" not in dfp.columns and "country_primary" in dfp.columns:
                dfp = dfp.rename(columns={"country_primary": "country"})
            dfp = dfp[dfp["country"].isin(sel_c)]
            ch = alt.Chart(dfp).mark_line().encode(
                x=x_year_axis("Year"),
                y=alt.Y("prevalence:Q", title="Prevalence"),
                color=alt.Color("country:N", title="Country", scale=alt.Scale(range[DISTINCT_PALETTE])),
                strokeDash=alt.StrokeDash("corpus:N", title="Corpus"),
                tooltip=["pub_year","country","corpus","prevalence"],
            ).properties(height=360)
            st.altair_chart(ch, use_container_width=True)
        elif not sel_c:
            st.info("Select some countries to plot annual prevalence.")
        else:
            st.info("Country CSVs not found for either corpus.")

        st.markdown("**Aggregated-GLM trends (OR per year)**")
        dfs_or = [d for d in load_df_all(key_or).values() if d is not None]
        if dfs_or:
            df_or = pd.concat(dfs_or, ignore_index=True)
            st.dataframe(df_or.sort_values(["corpus","p","N_total"]).reset_index(drop=True))
        else:
            st.info("Country ORs CSVs not found.")

# ──────────────────────────────────────────────────────────────────────────────
# LEXEMES
# ──────────────────────────────────────────────────────────────────────────────
with T_LEX:
    st.subheader("Lexemes (CORE set) over time")
    if SHOW_EXPLAINERS:
        st.info(
            "Trends for individual lexemes. In compare mode, lines are colored by lexeme and **dashed by corpus**."
        )

    dfs_lex = [d for d in load_df_all("lexemes").values() if d is not None]
    if not dfs_lex:
        st.info("Lexeme-level CSV not found.")
    else:
        def norm_target(x: object) -> str:
            s = str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
            if s in {"any", "union", "anyunion", "overall", "both", "all"}:
                return "any"
            if s in {"title", "titles"}:
                return "title"
            if s in {"abstract", "abstracts", "abs"}:
                return "abstract"
            return s

        df_lex_all = pd.concat(dfs_lex, ignore_index=True)
        if "target" in df_lex_all.columns:
            df_lex_all["_slice"] = df_lex_all["target"].map(norm_target)
            present = [s for s in ["any", "title", "abstract"] if (df_lex_all["_slice"] == s).any()]
            label_map = {"any": "Any (union)", "title": "Title", "abstract": "Abstract"}
            if present:
                choice_label = st.radio("Slice", [label_map[s] for s in present], horizontal=True, key="lex_slice")
                chosen = {v: k for k, v in label_map.items()}[choice_label]
            else:
                opts = sorted(df_lex_all["_slice"].dropna().unique().tolist())
                chosen = st.selectbox("Slice", opts, key="lex_slice_other")
            df_lex_all = df_lex_all[df_lex_all["_slice"] == chosen].copy()

        if df_lex_all.duplicated(["pub_year", "lexeme", "corpus"]).any():
            df_lex_all = (
                df_lex_all.groupby(["pub_year", "lexeme", "corpus"], as_index=False)["prevalence"]
                .mean()
                .sort_values(["lexeme", "pub_year", "corpus"])
            )

        lex_list = sorted(df_lex_all.get("lexeme", pd.Series(dtype=str)).dropna().unique().tolist())
        default_lex = lex_list[:6]
        sel_lex = st.multiselect("Lexemes", lex_list, default=default_lex, key="lex_sel")

        if sel_lex:
            dfp = df_lex_all[df_lex_all["lexeme"].isin(sel_lex)].copy()
            if _HAS_ALTAIR and {"prevalence"}.issubset(dfp.columns):
                ch = alt.Chart(dfp).mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("prevalence:Q", title="Prevalence"),
                    color=alt.Color("lexeme:N", scale=alt.Scale(range=DISTINCT_PALETTE)),
                    strokeDash=alt.StrokeDash("corpus:N", title="Corpus"),
                    tooltip=["pub_year","lexeme","corpus","prevalence"],
                ).properties(height=360)
                st.altair_chart(ch, use_container_width=True)
            else:
                st.dataframe(dfp)
        else:
            st.info("Select one or more lexemes to visualize.")

# ──────────────────────────────────────────────────────────────────────────────
# CONTEXTUAL (Cell-6)
# ──────────────────────────────────────────────────────────────────────────────
with T_MS:
    st.subheader("Contextual metaphor share (KWIC→LLM)")
    if SHOW_EXPLAINERS:
        st.info(
            "Among texts that contain a war-term, the *metaphor share* is the fraction judged metaphorical by an LLM. "
            "In compare mode, **dashed by corpus**."
        )

    colA, colB = st.columns([1, 1])

    # By year
    with colA:
        dfs_y = [d for d in load_df_all("mshare_year").values() if d is not None]
        if not dfs_y:
            st.info("metaphor_share_by_year_union.csv not found.")
        elif not is_compare:
            df = dfs_y[0]
            if {"pub_year","share_metaphor"}.issubset(df.columns):
                st.markdown("**By year**")
                line_with_band(df, y="share_metaphor", lo="lo95", hi="hi95", title="Metaphor share")
        else:
            if _HAS_ALTAIR:
                df_all = pd.concat(dfs_y, ignore_index=True)
                st.markdown("**By year — PubMed vs OpenAlex**")
                base = alt.Chart(df_all)
                band = None
                if {"lo95","hi95"}.issubset(df_all.columns):
                    band = base.mark_area(opacity=0.15).encode(
                        x=alt.X("pub_year:Q"), y="lo95:Q", y2="hi95:Q",
                        color=alt.Color("corpus:N", legend=None), detail="corpus:N"
                    )
                line = base.mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("share_metaphor:Q", title="Metaphor share"),
                    color=alt.Color("corpus:N", title="Corpus"),
                    strokeDash=alt.StrokeDash("corpus:N", legend=None),
                    tooltip=["pub_year","corpus","share_metaphor","n_hits","n_metaphor"],
                )
                st.altair_chart(((band + line) if band is not None else line).properties(height=340),
                                use_container_width=True)
            else:
                st.dataframe(pd.concat(dfs_y, ignore_index=True))

    # By doc type
    with colB:
        dfs_dt = [d for d in load_df_all("mshare_dt").values() if d is not None]
        if not dfs_dt:
            st.info("metaphor_share_by_doctype_union.csv not found.")
        else:
            df_all = pd.concat(dfs_dt, ignore_index=True)
            _clean = _drop_unknown(df_all, ["doc_type"])
            df_all = _clean if _clean is not None else df_all  # ← fixed (no 'or' with DF)
            if {"pub_year", "doc_type", "share_metaphor"}.issubset(df_all.columns):
                st.markdown("**By doc type**")
                if "n_hits" in df_all.columns:
                    top_dt = (df_all.groupby("doc_type")["n_hits"].sum()
                              .sort_values(ascending=False).head(6).index.tolist())
                else:
                    top_dt = df_all["doc_type"].dropna().unique().tolist()[:6]
                sel_dt = st.multiselect(
                    "Doc types",
                    sorted(df_all["doc_type"].dropna().unique().tolist()),
                    default=top_dt,
                    key="ms_dt_sel",
                )
                if sel_dt:
                    dfp = df_all[df_all["doc_type"].isin(sel_dt)]
                    if _HAS_ALTAIR:
                        ch = alt.Chart(dfp).mark_line().encode(
                            x=x_year_axis("Year"),
                            y=alt.Y("share_metaphor:Q", title="Metaphor share"),
                            color=alt.Color("doc_type:N", scale=alt.Scale(range=DISTINCT_PALETTE)),
                            strokeDash=alt.StrokeDash("corpus:N", title="Corpus"),
                            tooltip=["pub_year","doc_type","corpus","share_metaphor","n_hits","n_metaphor"],
                        ).properties(height=340)
                        st.altair_chart(ch, use_container_width=True)
                    else:
                        st.dataframe(dfp)

    # By country
    st.markdown("**By country**")
    dfs_cty = [d for d in load_df_all("mshare_cty").values() if d is not None]
    if not dfs_cty:
        st.info("metaphor_share_by_country_union.csv not found.")
    else:
        df_all = pd.concat(dfs_cty, ignore_index=True)
        _clean = _drop_unknown(df_all, ["country"])
        df_all = _clean if _clean is not None else df_all  # ← fixed (no 'or' with DF)
        if _HAS_ALTAIR and {"country","share_metaphor"}.issubset(df_all.columns):
            if "n_hits" in df_all.columns:
                top_cty = (df_all.groupby("country")["n_hits"].sum()
                           .sort_values(ascending=False).head(8).index.tolist())
            else:
                top_cty = df_all["country"].dropna().unique().tolist()[:8]
            sel_cty = st.multiselect(
                "Countries",
                sorted(df_all["country"].dropna().unique().tolist()),
                default=top_cty,
                key="ms_cty_sel",
            )
            if sel_cty:
                dfp = df_all[df_all["country"].isin(sel_cty)]
                ch = alt.Chart(dfp).mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("share_metaphor:Q", title="Metaphor share"),
                    color=alt.Color("country:N", scale=alt.Scale(range=DISTINCT_PALETTE)),
                    strokeDash=alt.StrokeDash("corpus:N", title="Corpus"),
                    tooltip=["pub_year","country","corpus","share_metaphor","n_hits","n_metaphor"],
                ).properties(height=360)
                st.altair_chart(ch, use_container_width=True)
        else:
            st.dataframe(df_all)

# ──────────────────────────────────────────────────────────────────────────────
# GLM COEFFICIENTS (Cell-3)
# ──────────────────────────────────────────────────────────────────────────────
with T_GLM:
    st.subheader("Aggregated GLM coefficients")
    if SHOW_EXPLAINERS:
        st.info(
            "Coefficients from your logistic GLM. In compare mode, the table includes a **corpus** column; "
            "OR transforms are shown per corpus."
        )

    which = st.radio("Slice", ["Any (union)", "Title", "Abstract"], horizontal=True, key="glm_slice")
    key_coef = {"Any (union)": "coef_any", "Title": "coef_title", "Abstract": "coef_abs"}[which]

    if not is_compare:
        df_coef = load_df_for(key_coef, active_corpora[0])
        if df_coef is None:
            st.info("Coefficient CSV not found for this slice.")
        else:
            st.dataframe(df_coef)
            if {"beta","se"}.issubset(df_coef.columns):
                try:
                    import numpy as np
                    tmp = df_coef.copy()
                    tmp["OR"] = np.exp(tmp["beta"])
                    tmp["OR_lo95"] = np.exp(tmp["beta"] - 1.959964*tmp["se"])
                    tmp["OR_hi95"] = np.exp(tmp["beta"] + 1.959964*tmp["se"])
                    st.dataframe(tmp[["param","OR","OR_lo95","OR_hi95","p","corpus"]])
                except Exception:
                    pass
    else:
        dfs = [d for d in load_df_all(key_coef).values() if d is not None]
        if dfs:
            dfc = pd.concat(dfs, ignore_index=True)
            st.dataframe(dfc)
            if {"beta","se"}.issubset(dfc.columns):
                try:
                    import numpy as np
                    tmp = dfc.copy()
                    tmp["OR"] = np.exp(tmp["beta"])
                    tmp["OR_lo95"] = np.exp(tmp["beta"] - 1.959964*tmp["se"])
                    tmp["OR_hi95"] = np.exp(tmp["beta"] + 1.959964*tmp["se"])
                    st.dataframe(tmp[["param","OR","OR_lo95","OR_hi95","p","corpus"]])
                except Exception:
                    pass
        else:
            st.info("Coefficient CSVs not found for either corpus.")

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOADS
# ──────────────────────────────────────────────────────────────────────────────
with T_DL:
    st.subheader("Quick downloads")
    if SHOW_EXPLAINERS:
        st.info(
            "Download the exact CSVs this app is reading. "
            "In compare mode, you’ll see separate sections per corpus."
        )

    if not is_compare:
        corpus = active_corpora[0]
        files_here = list_csvs(CORPUS_DIRS[corpus])
        if not files_here:
            st.info("No CSVs found in this directory.")
        else:
            for name, path in sorted(files_here.items()):
                with open(path, "rb") as fh:
                    st.download_button(
                        label=f"Download {name}",
                        data=fh.read(),
                        file_name=name,
                        mime="text/csv",
                        key=f"dl_{corpus}_{name}",
                    )
    else:
        for corpus in ["PubMed", "OpenAlex"]:
            st.markdown(f"**{corpus} files**")
            files_here = list_csvs(CORPUS_DIRS[corpus])
            if not files_here:
                st.info(f"No CSVs found in {corpus} directory.")
            else:
                for name, path in sorted(files_here.items()):
                    with open(path, "rb") as fh:
                        st.download_button(
                            label=f"[{corpus}] {name}",
                            data=fh.read(),
                            file_name=f"{corpus.lower()}_{name}",
                            mime="text/csv",
                            key=f"dl_{corpus}_{name}",
                        )

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="small">Copyright reserved by Complex Systems Lab @ Penn</div>
    """,
    unsafe_allow_html=True,
)
