SHELL_HTML = """
<style>
:root {
    --olive-ink: #202620;
    --olive-muted: #6b746d;
    --olive-line: rgba(40, 54, 44, 0.13);
    --olive-panel: rgba(255, 255, 255, 0.92);
    --olive-wash: #f6f7f3;
    --olive-accent: #526c57;
    --olive-accent-soft: rgba(82, 108, 87, 0.10);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% -10%, rgba(89, 113, 93, 0.08), transparent 26rem),
        linear-gradient(180deg, #fbfcf9 0%, #f7f8f5 100%);
    color: var(--olive-ink);
}

[data-testid="stHeader"] {
    background: rgba(251, 252, 249, 0.82);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1580px;
    padding-top: 1.45rem;
    padding-bottom: 4rem;
}

.olive-head {
    border: 1px solid var(--olive-line);
    border-radius: 24px;
    padding: 26px 30px 24px;
    margin: 0 0 16px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(244,247,241,0.94));
    box-shadow: 0 10px 30px rgba(42, 51, 44, 0.045);
}

.olive-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--olive-accent);
    font-weight: 750;
    margin-bottom: 8px;
}

.olive-title {
    font-size: clamp(2.0rem, 3vw, 3.15rem);
    letter-spacing: -0.045em;
    line-height: 1.02;
    font-weight: 760;
    color: var(--olive-ink);
    margin: 0;
}

.olive-subtitle {
    max-width: 920px;
    margin-top: 11px;
    font-size: 0.98rem;
    line-height: 1.55;
    color: var(--olive-muted);
}

.olive-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.olive-badge {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--olive-line);
    border-radius: 999px;
    padding: 6px 10px;
    background: rgba(255,255,255,0.72);
    color: #4f5a52;
    font-size: 0.74rem;
    font-weight: 650;
}

.olive-badge.measured {
    background: rgba(82, 108, 87, 0.10);
    color: #3f5c46;
}

.olive-badge.derived {
    background: rgba(94, 108, 127, 0.09);
    color: #4f6075;
}

.olive-badge.inferred {
    background: rgba(139, 107, 69, 0.09);
    color: #765b3e;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--olive-line) !important;
    border-radius: 18px !important;
    background: var(--olive-panel);
    box-shadow: 0 5px 18px rgba(44, 53, 46, 0.025);
}

[data-testid="stMetric"] {
    border: 1px solid var(--olive-line);
    border-radius: 16px;
    padding: 13px 15px;
    background: rgba(255,255,255,0.78);
    min-height: 92px;
}

[data-testid="stMetricLabel"] {
    color: var(--olive-muted);
}

[data-testid="stMetricValue"] {
    color: var(--olive-ink);
    letter-spacing: -0.025em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.20rem;
    overflow-x: auto;
    scrollbar-width: thin;
    border-bottom: 1px solid var(--olive-line);
    padding: 0 0 4px;
    margin-top: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 43px;
    border-radius: 10px 10px 0 0;
    padding-left: 13px;
    padding-right: 13px;
    color: var(--olive-muted);
    background: transparent;
    white-space: nowrap;
    font-weight: 590;
}

.stTabs [aria-selected="true"] {
    color: var(--olive-ink) !important;
    background: var(--olive-accent-soft) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--olive-accent) !important;
    height: 2px;
}

[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    border-radius: 11px !important;
}

.stButton > button,
[data-testid="stPageLink"] a {
    border-radius: 11px !important;
}

hr {
    border-color: var(--olive-line) !important;
}

h1, h2, h3, h4 {
    letter-spacing: -0.025em;
}

[data-testid="stCaptionContainer"] {
    color: var(--olive-muted);
}

[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>

<div class="olive-head">
  <div class="olive-kicker">XRF CHEMICAL CELL ANALYSIS PLATFORM</div>
  <div class="olive-title">SOURDOUGH</div>
  <div class="olive-subtitle">
        <strong>Synchrotron Observations Using Resolved Distributions Of Uptake,
        Geometry &amp; Heterogeneity</strong><br>
        XRF chemical imaging and cellular analysis.
      </div>
  <div class="olive-badges">
    <span class="olive-badge measured">● MEASURED · X/Y + XRF</span>
    <span class="olive-badge derived">◆ DERIVED · gradients + Hessian + features</span>
    <span class="olive-badge inferred">◇ INFERRED · model-based Z geometry</span>
  </div>
</div>
"""
