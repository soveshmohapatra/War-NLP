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
from typing import Dict, Optional

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

def _drop_unknown(df: Optional[pd.DataFrame], cols: list[str]) -> Optional[pd.DataFrame]:
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
source = st.sidebar.radio("Corpus", ["PubMed", "OpenAlex"], index=0, horizontal=True, key="src")

# Hard clamp year inputs to 2010–2025 (cannot go outside)
min_year = int(st.sidebar.number_input(
    "Min year", value=2010, step=1, min_value=2010, max_value=2025, key="ymin"
))
max_year = int(st.sidebar.number_input(
    "Max year", value=2025, step=1, min_value=2010, max_value=2025, key="ymax"
))
# Keep range sensible if user flips them
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

# Fixed directories (no sidebar inputs)
pubmed_dir = DEFAULT_PUBMED_DIR
openalex_dir = DEFAULT_OPENALEX_DIR
base_dir = pubmed_dir if source == "PubMed" else openalex_dir

available = list_csvs(base_dir)

# ──────────────────────────────────────────────────────────────────────────────
# Map of expected files → purposes
# ──────────────────────────────────────────────────────────────────────────────
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
            "**How to read.** An upward line = more frequent mentions. Shaded bands are 95% CIs (if provided). "
            "Use the year filter in the sidebar to focus the time window."
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

# ──────────────────────────────────────────────────────────────────────────────
# TITLE vs ABSTRACT
# ──────────────────────────────────────────────────────────────────────────────
with T_TVA:
    st.subheader("Where do war-terms show up — titles or abstracts?")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** A direct comparison of where war-terms appear: in paper titles vs. in abstracts.\n\n"
            "**Why it matters.** Title mentions often signal framing or emphasis; abstract mentions reflect fuller discussion. "
            "Divergence between the lines can indicate shifts in how prominently the topic is presented."
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
# DOC TYPES
# ──────────────────────────────────────────────────────────────────────────────
with T_DT:
    st.subheader("Document types")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Prevalence by publication type (e.g., Article, Review, Editorial). "
            "The shaded band is a 95% confidence interval (if available).\n\n"
            "**How to use.** Pick the slice (Title/Abstract), choose the top-K types on the left, "
            "and select which types to plot. The table below shows aggregated GLM odds-ratio trends per year."
        )

    left, right = st.columns([1, 1])

    with left:
        which = st.radio("Slice", ["Title", "Abstract"], horizontal=True, key="dt_slice")
        p_sizes = resolve_file("dt_sizes_title" if which == "Title" else "dt_sizes_abs")
        p_yearly = resolve_file("dt_title" if which == "Title" else "dt_abs")
        p_or = resolve_file("dt_or_title" if which == "Title" else "dt_or_abs")

        df_sizes = _drop_unknown(load_csv_optional(p_sizes) if p_sizes else None, ["doc_type"])
        df_year  = _drop_unknown(_clip_years(load_csv_optional(p_yearly) if p_yearly else None), ["doc_type_major","doc_type"])
        df_or    = _drop_unknown(load_csv_optional(p_or) if p_or else None, ["doc_type","doc_type_major"])

        if df_sizes is not None and "doc_type" in df_sizes.columns:
            df_sizes = df_sizes.sort_values("n_docs", ascending=False)
            topk = st.slider("Show top K by total N", 3, 12, 6, key="dt_topk")
            default_list = df_sizes["doc_type"].head(topk).tolist()
            sel = st.multiselect(
                "Doc types",
                df_sizes["doc_type"].head(20).tolist(),
                default=default_list,
                key=f"dt_sel_{which}_{topk}",
            )
        else:
            sel = []

    with right:
        if df_year is not None and sel:
            df_plot = df_year[df_year.get("doc_type_major", df_year.get("doc_type")).isin(sel)].copy()
            if "doc_type" not in df_plot.columns and "doc_type_major" in df_plot.columns:
                df_plot = df_plot.rename(columns={"doc_type_major": "doc_type"})

            if _HAS_ALTAIR and {"prevalence", "prev_lo95", "prev_hi95"}.issubset(df_plot.columns):
                base = alt.Chart(df_plot)
                color_dt = alt.Color("doc_type:N", title="Doc type", scale=alt.Scale(range=DISTINCT_PALETTE))
                line = base.mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("prevalence:Q", title="Prevalence"),
                    color=color_dt,
                )
                band = base.mark_area(opacity=0.18).encode(
                    x=alt.X("pub_year:Q"),
                    y="prev_lo95:Q",
                    y2="prev_hi95:Q",
                    color=alt.Color("doc_type:N", scale=alt.Scale(range=DISTINCT_PALETTE), legend=None),
                )
                st.altair_chart((line + band).properties(height=360), use_container_width=True)
            else:
                st.dataframe(df_plot)
        else:
            st.info("Select some doc types on the left to plot annual prevalence.")

    st.markdown("**Aggregated-GLM trends (OR per year)**")
    if df_or is not None:
        df_or = df_or.copy()
        if "doc_type_major" in df_or.columns and "doc_type" not in df_or.columns:
            df_or = df_or.rename(columns={"doc_type_major": "doc_type"})
        st.dataframe(df_or.sort_values(["p", "N_total"]).reset_index(drop=True))
    else:
        st.info("Doc-type ORs CSV not found.")

# ──────────────────────────────────────────────────────────────────────────────
# COUNTRIES
# ──────────────────────────────────────────────────────────────────────────────
with T_CTY:
    st.subheader("Countries")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Prevalence by country (based on author affiliation; often the primary/first affiliation). "
            "Lines are annual prevalence; shaded bands show 95% CIs when available.\n\n"
            "**Notes.** Country assignment depends on your preprocessing. Use the multiselect to compare countries with enough data."
        )

    c_slice = st.radio("Slice", ["Any (union)", "Title", "Abstract"], horizontal=True, key="cty_slice")

    if c_slice == "Any (union)":
        p_cty = resolve_file("cty_any");   p_or = resolve_file("cty_or_any")
    elif c_slice == "Title":
        p_cty = resolve_file("cty_title"); p_or = resolve_file("cty_or_title")
    else:
        p_cty = resolve_file("cty_abs");   p_or = resolve_file("cty_or_abs")
    p_sizes = resolve_file("cty_sizes")

    df_cty   = _drop_unknown(_clip_years(load_csv_optional(p_cty) if p_cty else None), ["country","country_primary"])
    df_sizes = _drop_unknown(load_csv_optional(p_sizes) if p_sizes else None, ["country"])
    df_or    = _drop_unknown(load_csv_optional(p_or) if p_or else None, ["country"])

    if df_sizes is not None and "country" in df_sizes.columns:
        df_sizes = df_sizes.sort_values("n_docs", ascending=False)
        default_c = df_sizes["country"].head(8).tolist()
        sel_c = st.multiselect("Countries", df_sizes["country"].tolist(), default=default_c, key="cty_sel")
    else:
        sel_c = []

    if df_cty is not None and sel_c:
        dfp = df_cty[df_cty.get("country_primary", df_cty.get("country")).isin(sel_c)].copy()
        if "country" not in dfp.columns and "country_primary" in dfp.columns:
            dfp = dfp.rename(columns={"country_primary": "country"})

        if _HAS_ALTAIR and {"prevalence", "prev_lo95", "prev_hi95"}.issubset(dfp.columns):
            base = alt.Chart(dfp)
            line = base.mark_line().encode(
                x=x_year_axis("Year"),
                y=alt.Y("prevalence:Q", title="Prevalence"),
                color=alt.Color("country:N", title="Country", scale=alt.Scale(range=DISTINCT_PALETTE)),
            )
            band = base.mark_area(opacity=0.18).encode(
                x=alt.X("pub_year:Q"),
                y="prev_lo95:Q",
                y2="prev_hi95:Q",
                color=alt.Color("country:N", scale=alt.Scale(range=DISTINCT_PALETTE), legend=None),
            )
            st.altair_chart((line + band).properties(height=360), use_container_width=True)
        else:
            st.dataframe(dfp)
    else:
        st.info("Select some countries to plot annual prevalence.")

    st.markdown("**Aggregated-GLM trends (OR per year)**")
    if df_or is not None:
        st.dataframe(df_or.sort_values(["p", "N_total"]).reset_index(drop=True))
    else:
        st.info("Country ORs CSV not found.")

# ──────────────────────────────────────────────────────────────────────────────
# LEXEMES
# ──────────────────────────────────────────────────────────────────────────────
with T_LEX:
    st.subheader("Lexemes (CORE set) over time")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Trends for individual words/phrases in the core ‘war’ set. "
            "Each line is one lexeme.\n\n"
            "**Tips.** Use *Slice* to choose Title/Abstract/Any. Select lexemes to compare. "
            "If multiple rows existed per (year, lexeme), they were averaged for smooth lines."
        )

    p_lex = resolve_file("lexemes")
    df_lex = _clip_years(load_csv_optional(p_lex) if p_lex else None)

    if df_lex is None:
        st.info("Lexeme-level CSV not found.")
    else:
        if "target" in df_lex.columns:
            def _norm_target(x: object) -> str:
                s = str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
                if s in {"any", "union", "anyunion", "overall", "both", "all"}:
                    return "any"
                if s in {"title", "titles"}:
                    return "title"
                if s in {"abstract", "abstracts", "abs"}:
                    return "abstract"
                return s
            df_lex["_slice"] = df_lex["target"].map(_norm_target)

            present = [s for s in ["any", "title", "abstract"] if (df_lex["_slice"] == s).any()]
            label_map = {"any": "Any (union)", "title": "Title", "abstract": "Abstract"}

            if present:
                choice_label = st.radio(
                    "Slice", [label_map[s] for s in present], horizontal=True, key="lex_slice"
                )
                chosen = {v: k for k, v in label_map.items()}[choice_label]
            else:
                opts = sorted(df_lex["_slice"].dropna().unique().tolist())
                chosen = st.selectbox("Slice", opts, key="lex_slice_other")

            df_lex = df_lex[df_lex["_slice"] == chosen].copy()

        if df_lex.duplicated(["pub_year", "lexeme"]).any():
            before = len(df_lex)
            df_lex = (
                df_lex.groupby(["pub_year", "lexeme"], as_index=False)["prevalence"]
                .mean()
                .sort_values(["lexeme", "pub_year"])
            )
            after = len(df_lex)
            st.caption(f"Collapsed duplicate rows per (year, lexeme): {before} → {after}")

        lex_list = sorted(df_lex.get("lexeme", pd.Series(dtype=str)).dropna().unique().tolist())
        default_lex = lex_list[:6]
        sel_lex = st.multiselect("Lexemes", lex_list, default=default_lex, key="lex_sel")

        if sel_lex:
            dfp = df_lex[df_lex["lexeme"].isin(sel_lex)].copy()
            if _HAS_ALTAIR and {"prevalence"}.issubset(dfp.columns):
                ch = alt.Chart(dfp).mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("prevalence:Q", title="Prevalence"),
                    color=alt.Color("lexeme:N", scale=alt.Scale(range=DISTINCT_PALETTE)),
                    tooltip=["pub_year", "lexeme", "prevalence"],
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
            "**What this shows.** Among texts that contain a war-term, the *metaphor share* is the fraction "
            "of usages judged metaphorical by an LLM reading KWIC snippets.\n\n"
            "**Panels.** Left: change by year. Right: breakdown by document type. Below: breakdown by country.\n\n"
            "**Notes.** Shares come from sampled hits (n_metaphor / n_hits); wider CIs imply fewer samples."
        )

    colA, colB = st.columns([1, 1])

    with colA:
        p_ms_y = resolve_file("mshare_year")
        df_ms_y = _clip_years(load_csv_optional(p_ms_y) if p_ms_y else None)
        if df_ms_y is not None and {"pub_year", "share_metaphor", "lo95", "hi95"}.issubset(df_ms_y.columns):
            st.markdown("**By year**")
            line_with_band(df_ms_y, y="share_metaphor", lo="lo95", hi="hi95", title="Metaphor share")
        else:
            st.info("metaphor_share_by_year_union.csv not found or missing cols.")

    with colB:
        p_ms_dt = resolve_file("mshare_dt")
        df_ms_dt = _drop_unknown(_clip_years(load_csv_optional(p_ms_dt) if p_ms_dt else None), ["doc_type"])
        if df_ms_dt is not None and {"pub_year", "doc_type", "share_metaphor"}.issubset(df_ms_dt.columns):
            st.markdown("**By doc type**")
            top_dt = (
                df_ms_dt.groupby("doc_type")["n_hits"].sum().sort_values(ascending=False).head(6).index.tolist()
                if "n_hits" in df_ms_dt.columns else df_ms_dt["doc_type"].dropna().unique()[:6].tolist()
            )
            sel_dt = st.multiselect(
                "Doc types",
                sorted(df_ms_dt["doc_type"].dropna().unique().tolist()),
                default=top_dt,
                key="ms_dt_sel",
            )
            if sel_dt:
                dfp = df_ms_dt[df_ms_dt["doc_type"].isin(sel_dt)]
                if _HAS_ALTAIR:
                    ch = alt.Chart(dfp).mark_line().encode(
                        x=x_year_axis("Year"),
                        y=alt.Y("share_metaphor:Q", title="Metaphor share"),
                        color=alt.Color("doc_type:N", scale=alt.Scale(range=DISTINCT_PALETTE)),
                        tooltip=["pub_year", "doc_type", "share_metaphor", "n_hits", "n_metaphor"],
                    ).properties(height=340)
                    st.altair_chart(ch, use_container_width=True)
                else:
                    st.dataframe(dfp)
        else:
            st.info("metaphor_share_by_doctype_union.csv not found or missing cols.")

    st.markdown("**By country**")
    p_ms_cty = resolve_file("mshare_cty")
    df_ms_cty = _drop_unknown(_clip_years(load_csv_optional(p_ms_cty) if p_ms_cty else None), ["country"])
    if df_ms_cty is not None and {"country", "share_metaphor"}.issubset(df_ms_cty.columns):
        if _HAS_ALTAIR:
            top_cty = (
                df_ms_cty.groupby("country")["n_hits"].sum().sort_values(ascending=False).head(8).index.tolist()
                if "n_hits" in df_ms_cty.columns else df_ms_cty["country"].dropna().unique()[:8].tolist()
            )
            sel_cty = st.multiselect(
                "Countries",
                sorted(df_ms_cty["country"].dropna().unique().tolist()),
                default=top_cty,
                key="ms_cty_sel",
            )
            if sel_cty:
                dfp = df_ms_cty[df_ms_cty["country"].isin(sel_cty)]
                ch = alt.Chart(dfp).mark_line().encode(
                    x=x_year_axis("Year"),
                    y=alt.Y("share_metaphor:Q", title="Metaphor share"),
                    color=alt.Color("country:N", scale=alt.Scale(range=DISTINCT_PALETTE)),
                    tooltip=["pub_year", "country", "share_metaphor", "n_hits", "n_metaphor"],
                ).properties(height=360)
                st.altair_chart(ch, use_container_width=True)
        else:
            st.dataframe(df_ms_cty)
    else:
        st.info("metaphor_share_by_country_union.csv not found or missing cols.")

# ──────────────────────────────────────────────────────────────────────────────
# GLM COEFFICIENTS (Cell-3)
# ──────────────────────────────────────────────────────────────────────────────
with T_GLM:
    st.subheader("Aggregated GLM coefficients")
    if SHOW_EXPLAINERS:
        st.info(
            "**What this shows.** Coefficients from a logistic GLM where the outcome is whether a document contains a war-term. "
            "`beta` is a log-odds coefficient; `OR = exp(beta)` is the multiplicative change in odds.\n\n"
            "**How to read.** OR > 1 indicates higher odds; OR < 1 lower odds. "
            "The year term (if present) reflects per-year change after controls (per your model spec)."
        )

    which = st.radio("Slice", ["Any (union)", "Title", "Abstract"], horizontal=True, key="glm_slice")
    if which == "Any (union)":
        p_coef = resolve_file("coef_any")
    elif which == "Title":
        p_coef = resolve_file("coef_title")
    else:
        p_coef = resolve_file("coef_abs")

    df_coef = load_csv_optional(p_coef) if p_coef else None
    if df_coef is None:
        st.info("Coefficient CSV not found for this slice.")
    else:
        st.dataframe(df_coef)
        if {"beta","se"}.issubset(df_coef.columns):
            st.caption("Odds-ratio transformations:")
            try:
                import numpy as np
                tmp = df_coef.copy()
                tmp["OR"] = np.exp(tmp["beta"])
                tmp["OR_lo95"] = np.exp(tmp["beta"] - 1.959964*tmp["se"])
                tmp["OR_hi95"] = np.exp(tmp["beta"] + 1.959964*tmp["se"])
                st.dataframe(tmp[["param","OR","OR_lo95","OR_hi95","p"]])
            except Exception:
                pass

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOADS
# ──────────────────────────────────────────────────────────────────────────────
with T_DL:
    st.subheader("Quick downloads")
    if SHOW_EXPLAINERS:
        st.info(
            "Download the exact CSVs this app is reading from the selected data directory. "
            "Use these for replication, external plotting, or sharing."
        )

    files_here = list_csvs(base_dir)
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
                    key=f"dl_{name}",
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
