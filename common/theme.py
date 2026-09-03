"""Shared look-and-feel for every Streamlit app in this portfolio.

Import and call `apply(icon, title)` once at the top of each app's
`main()` to get a consistent color palette, typography, CRISP-DM phase
badges, and a footer linking back to the monorepo. Keeping this in one
module (instead of copy-pasting CSS into every app) is the one bit of
shared infrastructure across otherwise-independent projects.
"""
import streamlit as st

PALETTE = {
    "bg": "#0b1220",
    "panel": "#111a2e",
    "panel_alt": "#0f1729",
    "border": "#243050",
    "text": "#e6ebf5",
    "muted": "#93a1c2",
    "accent": "#6ee7f2",
    "accent2": "#a78bfa",
    "good": "#34d399",
    "warn": "#fbbf24",
    "bad": "#fb7185",
}

CRISP_DM_PHASES = [
    ("1. Business Understanding", "🎯"),
    ("2. Data Understanding", "🔎"),
    ("3. Data Preparation", "🧹"),
    ("4. Modeling", "🧠"),
    ("5. Evaluation", "📏"),
    ("6. Deployment", "🚀"),
]

REPO_URL = "https://github.com/Akshata4/data-science-experiments"


def apply(icon: str, title: str, layout: str = "wide") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout=layout,
                        initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PALETTE['bg']}; color: {PALETTE['text']}; }}
        section[data-testid="stSidebar"] {{
            background: {PALETTE['panel_alt']};
            border-right: 1px solid {PALETTE['border']};
        }}
        h1, h2, h3, h4 {{ color: {PALETTE['text']} !important; font-weight: 700 !important; }}
        p, li, span, label {{ color: {PALETTE['text']}; }}
        .ds-hero {{
            background: linear-gradient(135deg, {PALETTE['panel']} 0%, {PALETTE['panel_alt']} 100%);
            border: 1px solid {PALETTE['border']};
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }}
        .ds-hero h1 {{ margin: 0 0 .25rem 0; font-size: 1.9rem; }}
        .ds-hero p {{ color: {PALETTE['muted']}; margin: 0; font-size: 0.98rem; }}
        .ds-badgebar {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem; }}
        .ds-badge {{
            background: {PALETTE['panel_alt']};
            border: 1px solid {PALETTE['border']};
            color: {PALETTE['accent']};
            border-radius: 999px;
            padding: .25rem .75rem;
            font-size: .78rem;
            white-space: nowrap;
        }}
        .ds-card {{
            background: {PALETTE['panel']};
            border: 1px solid {PALETTE['border']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            height: 100%;
        }}
        .ds-metric-label {{ color: {PALETTE['muted']}; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }}
        .ds-metric-value {{ font-size: 1.6rem; font-weight: 700; color: {PALETTE['accent']}; }}
        .ds-footer {{
            margin-top: 2.5rem; padding-top: 1rem;
            border-top: 1px solid {PALETTE['border']};
            color: {PALETTE['muted']}; font-size: .85rem;
        }}
        div[data-testid="stMetric"] {{
            background: {PALETTE['panel']};
            border: 1px solid {PALETTE['border']};
            border-radius: 10px;
            padding: .6rem .8rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {PALETTE['panel']};
            border-radius: 8px 8px 0 0;
            padding: .5rem 1rem;
        }}
        code {{ color: {PALETTE['accent2']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(icon: str, title: str, subtitle: str, phase_labels=None) -> None:
    badges = "".join(f'<span class="ds-badge">{p}</span>' for p in (phase_labels or []))
    st.markdown(
        f"""
        <div class="ds-hero">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
            <div class="ds-badgebar">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 360):
    """Apply the dark portfolio theme to a Plotly figure in place and return it."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font_color=PALETTE["text"],
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])
    fig.update_yaxes(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])
    return fig


def footer(project_name: str, prompts_link: str = "PROMPTS.md") -> None:
    st.markdown(
        f"""
        <div class="ds-footer">
        Part of the <a href="{REPO_URL}" target="_blank">Data Science Experiments</a> portfolio
        &nbsp;•&nbsp; {project_name} &nbsp;•&nbsp;
        Built with <a href="{REPO_URL}" target="_blank">Claude Code</a> — see this project's
        <code>{prompts_link}</code> for the prompt log.
        </div>
        """,
        unsafe_allow_html=True,
    )
