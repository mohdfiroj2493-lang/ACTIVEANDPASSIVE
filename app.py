import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lateral Earth Pressure Calculator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.95rem; }

    .result-card {
        background: #f0f7ff;
        border-left: 5px solid #2d6a9f;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .result-card h4 { margin: 0 0 8px; color: #1a3a5c; }

    .method-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .rankine  { background:#dbeafe; color:#1e40af; }
    .coulomb  { background:#dcfce7; color:#166534; }
    .atrest   { background:#fef9c3; color:#854d0e; }

    .formula-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: monospace;
        font-size: 0.88rem;
        margin: 6px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏗️ Lateral Earth Pressure Calculator</h1>
  <p>Rankine · Coulomb · At-Rest · Surcharge Pressure Analysis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar – Input Parameters
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input Parameters")

    st.subheader("🪨 Soil Properties")
    gamma = st.number_input("Unit Weight γ (kN/m³)", 10.0, 25.0, 18.0, 0.5,
                            help="Bulk unit weight of backfill soil")
    phi_deg = st.number_input("Friction Angle φ (°)", 0.0, 45.0, 30.0, 1.0,
                              help="Effective internal friction angle")
    c = st.number_input("Cohesion c (kPa)", 0.0, 100.0, 0.0, 1.0,
                        help="Effective cohesion (use 0 for granular soils)")
    delta_deg = st.number_input("Wall Friction δ (°)", 0.0, 35.0, 15.0, 1.0,
                                help="Wall–soil interface friction angle (Coulomb only)")

    st.subheader("🧱 Wall Geometry")
    H = st.number_input("Wall Height H (m)", 1.0, 30.0, 6.0, 0.5)
    alpha_deg = st.number_input("Wall Inclination α (°)", -20.0, 20.0, 0.0, 1.0,
                                help="Angle of wall face from vertical (+ve = leaning into fill)")
    beta_deg = st.number_input("Backfill Slope β (°)", 0.0, 30.0, 0.0, 1.0,
                               help="Slope of backfill surface")

    st.subheader("📦 Surcharge")
    surcharge_type = st.selectbox("Surcharge Type",
                                  ["None", "Uniform", "Line Load", "Point Load"])
    q_uniform = 0.0
    Q_line = 0.0; x_line = 1.0
    Q_point = 0.0; x_point = 1.0; z_point = 1.0

    if surcharge_type == "Uniform":
        q_uniform = st.number_input("Uniform Load q (kPa)", 0.0, 500.0, 10.0, 5.0)
    elif surcharge_type == "Line Load":
        Q_line  = st.number_input("Line Load Q (kN/m)", 0.0, 1000.0, 50.0, 10.0)
        x_line  = st.number_input("Distance from wall x (m)", 0.1, 20.0, 2.0, 0.1)
    elif surcharge_type == "Point Load":
        Q_point = st.number_input("Point Load Q (kN)", 0.0, 5000.0, 200.0, 50.0)
        x_point = st.number_input("Distance from wall x (m)", 0.1, 20.0, 2.0, 0.1)
        z_point = st.number_input("Depth of application z (m)", 0.0, 30.0, 0.0, 0.5)

    st.subheader("📊 Display Options")
    n_layers = st.slider("Pressure diagram points", 20, 200, 50)
    show_cohesion = st.checkbox("Show cohesion effects", True)
    show_net = st.checkbox("Show net active pressure", True)

# ─────────────────────────────────────────────
# Core calculations
# ─────────────────────────────────────────────
phi = np.radians(phi_deg)
delta = np.radians(delta_deg)
alpha = np.radians(alpha_deg)
beta = np.radians(beta_deg)

# ── Rankine coefficients ──────────────────────
def rankine_Ka(phi, beta):
    if beta == 0:
        return np.tan(np.pi/4 - phi/2)**2
    num = np.cos(beta) - np.sqrt(np.cos(beta)**2 - np.cos(phi)**2)
    den = np.cos(beta) + np.sqrt(np.cos(beta)**2 - np.cos(phi)**2)
    return np.cos(beta) * (num / den)

def rankine_Kp(phi, beta):
    if beta == 0:
        return np.tan(np.pi/4 + phi/2)**2
    num = np.cos(beta) + np.sqrt(np.cos(beta)**2 - np.cos(phi)**2)
    den = np.cos(beta) - np.sqrt(np.cos(beta)**2 - np.cos(phi)**2)
    return np.cos(beta) * (num / den)

# ── Coulomb coefficients ─────────────────────
def coulomb_Ka(phi, delta, alpha, beta):
    try:
        num = np.cos(phi - alpha)**2
        t1 = np.cos(alpha)**2 * np.cos(delta + alpha)
        sin_term = np.sqrt(
            np.sin(phi + delta) * np.sin(phi - beta) /
            (np.cos(delta + alpha) * np.cos(alpha - beta))
        )
        den = t1 * (1 + sin_term)**2
        return num / den
    except Exception:
        return np.tan(np.pi/4 - phi/2)**2

def coulomb_Kp(phi, delta, alpha, beta):
    try:
        num = np.cos(phi + alpha)**2
        t1 = np.cos(alpha)**2 * np.cos(delta - alpha)
        sin_term = np.sqrt(
            np.sin(phi + delta) * np.sin(phi + beta) /
            (np.cos(delta - alpha) * np.cos(alpha - beta))
        )
        den = t1 * (1 - sin_term)**2
        return num / den
    except Exception:
        return np.tan(np.pi/4 + phi/2)**2

# ── At-rest coefficient ──────────────────────
def K0(phi, beta=0):
    return (1 - np.sin(phi)) * (1 + np.sin(beta))

Ka_rankine = rankine_Ka(phi, beta)
Kp_rankine = rankine_Kp(phi, beta)
Ka_coulomb = coulomb_Ka(phi, delta, alpha, beta)
Kp_coulomb = coulomb_Kp(phi, delta, alpha, beta)
K0_val     = K0(phi, beta)

# ── Depth array ──────────────────────────────
depths = np.linspace(0, H, n_layers)

def active_pressure(K, z, gamma, c, q_s=0):
    pa = K * gamma * z - 2 * c * np.sqrt(K) + K * q_s
    if not show_cohesion:
        pa = K * gamma * z + K * q_s
    return pa

def passive_pressure(K, z, gamma, c, q_s=0):
    return K * gamma * z + 2 * c * np.sqrt(K) + K * q_s

def atrest_pressure(K0, z, gamma, c=0, q_s=0):
    return K0 * gamma * z + K0 * q_s

# Surcharge lateral pressure profiles
def surcharge_pressure(z_arr, stype, q=0, Q=0, x=1, z0=0, K=1):
    """Returns lateral pressure increment at depths z_arr due to surcharge."""
    p = np.zeros_like(z_arr, dtype=float)
    if stype == "Uniform":
        p = K * q * np.ones_like(z_arr)
    elif stype == "Line Load":
        # Boussinesq line load (Terzaghi modified)
        for i, z in enumerate(z_arr):
            if z == 0:
                continue
            m = x / H
            n = z / H
            if m > 0.4:
                p[i] = (2 * Q / np.pi / H) * (m**2 * n / (m**2 + n**2)**2)
            else:
                p[i] = (0.203 * Q / H) * (n / (0.16 + n**2)**2)
    elif stype == "Point Load":
        # Boussinesq point load
        for i, z in enumerate(z_arr):
            if z == 0:
                continue
            R = np.sqrt(x**2 + z**2)
            m = x / H
            n = z / H
            if m > 0.4:
                p[i] = (3 * Q / (2 * np.pi * H**2)) * (m**2 * n**3 / (m**2 + n**2)**2.5)
            else:
                p[i] = (0.28 * Q / H**2) * (n**3 / (0.16 + n**2)**3)
    return p

# Compute pressures
pa_rankine = active_pressure(Ka_rankine, depths, gamma, c, q_uniform)
pp_rankine = passive_pressure(Kp_rankine, depths, gamma, c, q_uniform)

pa_coulomb = active_pressure(Ka_coulomb, depths, gamma, c, q_uniform)
pp_coulomb = passive_pressure(Kp_coulomb, depths, gamma, c, q_uniform)

pa_atrest  = atrest_pressure(K0_val, depths, gamma, c, q_uniform)

# Add surcharge for line/point loads
if surcharge_type in ["Line Load", "Point Load"]:
    sur_pa_r = surcharge_pressure(depths, surcharge_type, Q=Q_line or Q_point,
                                   x=x_line or x_point, K=Ka_rankine)
    sur_pa_c = surcharge_pressure(depths, surcharge_type, Q=Q_line or Q_point,
                                   x=x_line or x_point, K=Ka_coulomb)
    sur_pa_0 = surcharge_pressure(depths, surcharge_type, Q=Q_line or Q_point,
                                   x=x_line or x_point, K=K0_val)
    pa_rankine += sur_pa_r
    pa_coulomb += sur_pa_c
    pa_atrest  += sur_pa_0

# Net active (clip negatives for cohesive soils display)
pa_rankine_net = np.clip(pa_rankine, 0, None)
pa_coulomb_net = np.clip(pa_coulomb, 0, None)

# Resultant forces (trapz integration)
def resultant(p, z):
    F = np.trapz(p, z)
    if F <= 0:
        return 0.0, H / 3
    moment = np.trapz(p * (max(z) - z), z)
    h_bar = moment / F
    return F, h_bar

Fa_r, ha_r = resultant(pa_rankine_net, depths)
Fa_c, ha_c = resultant(pa_coulomb_net, depths)
Fa_0, ha_0 = resultant(pa_atrest,      depths)
Fp_r, hp_r = resultant(pp_rankine,     depths)
Fp_c, hp_c = resultant(pp_coulomb,     depths)

# ─────────────────────────────────────────────
# Tabs layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📐 Results & Diagrams", "📊 Comparison Charts", "📋 Formulas & Theory", "🔢 Detailed Tables"])

# ════════════════════════════════════════════
# TAB 1 – Results & Diagrams
# ════════════════════════════════════════════
with tab1:
    col_res, col_diag = st.columns([1, 1.6])

    with col_res:
        st.subheader("Pressure Coefficients")
        df_K = {
            "Method": ["Rankine", "Coulomb", "At-Rest"],
            "Ka":  [f"{Ka_rankine:.4f}", f"{Ka_coulomb:.4f}", f"{K0_val:.4f}"],
            "Kp":  [f"{Kp_rankine:.4f}", f"{Kp_coulomb:.4f}", "—"],
        }
        st.table(df_K)

        st.subheader("Active Pressure Resultants")

        def result_card(method, cls, F, h, K):
            st.markdown(f"""
            <div class="result-card">
              <span class="method-badge {cls}">{method}</span>
              <h4>F<sub>a</sub> = {F:.2f} kN/m</h4>
              <p style="margin:0;font-size:0.85rem;">
                Acts at <b>{h:.2f} m</b> from base &nbsp;|&nbsp; K = {K:.4f}
              </p>
            </div>""", unsafe_allow_html=True)

        result_card("Rankine", "rankine", Fa_r, ha_r, Ka_rankine)
        result_card("Coulomb", "coulomb", Fa_c, ha_c, Ka_coulomb)
        result_card("At-Rest", "atrest",  Fa_0, ha_0, K0_val)

        st.subheader("Passive Pressure Resultants")
        st.markdown(f"""
        <div class="result-card">
          <span class="method-badge rankine">Rankine</span>
          <h4>F<sub>p</sub> = {Fp_r:.2f} kN/m</h4>
          <p style="margin:0;font-size:0.85rem;">Acts at <b>{hp_r:.2f} m</b> from base</p>
        </div>
        <div class="result-card">
          <span class="method-badge coulomb">Coulomb</span>
          <h4>F<sub>p</sub> = {Fp_c:.2f} kN/m</h4>
          <p style="margin:0;font-size:0.85rem;">Acts at <b>{hp_c:.2f} m</b> from base</p>
        </div>""", unsafe_allow_html=True)

    # ── Pressure diagrams ──────────────────────
    with col_diag:
        fig, axes = plt.subplots(1, 3, figsize=(11, 7), sharey=True)
        fig.patch.set_facecolor('#f8fafc')

        configs = [
            ("Rankine",  pa_rankine_net, pp_rankine, "#3b82f6", "#f59e0b"),
            ("Coulomb",  pa_coulomb_net, pp_coulomb, "#22c55e", "#ef4444"),
            ("At-Rest",  pa_atrest,       pa_atrest,  "#a855f7", "#a855f7"),
        ]

        for ax, (title, pa, pp, c_a, c_p) in zip(axes, configs):
            ax.set_facecolor('#f8fafc')

            # Passive (right side)
            ax.fill_betweenx(depths, 0, pp, alpha=0.15, color=c_p)
            ax.plot(pp, depths, color=c_p, lw=2, label="Passive")

            # Active (left side)
            ax.fill_betweenx(depths, 0, -pa, alpha=0.2, color=c_a)
            ax.plot(-pa, depths, color=c_a, lw=2.2, label="Active")

            ax.axvline(0, color='#334155', lw=0.8)
            ax.invert_yaxis()
            ax.set_title(title, fontweight='bold', fontsize=12, color='#1a3a5c')
            ax.set_xlabel("Pressure (kPa)", fontsize=9)
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=8)

        axes[0].set_ylabel("Depth (m)", fontsize=10)
        fig.suptitle("Lateral Earth Pressure Diagrams", fontsize=13,
                     fontweight='bold', color='#1a3a5c', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Wall schematic ─────────────────────
        st.subheader("Wall Schematic")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor('#f8fafc')
        ax2.set_facecolor('#f8fafc')

        # Draw wall
        ax2.fill_betweenx([0, H], [-0.3, -0.3], [0, 0], color='#94a3b8', alpha=0.7)
        # Soil
        ax2.fill_betweenx([0, H], [0, 0], [max(pa_rankine_net)/gamma + 1]*2,
                          color='#d97706', alpha=0.12)

        # Pressure arrows
        scale = H / (max(pa_rankine_net) + 0.001) * 0.6
        for i in range(0, n_layers, n_layers//8):
            p = pa_rankine_net[i]
            ax2.annotate("", xy=(-0.1, depths[i]),
                         xytext=(p * scale + 0.1, depths[i]),
                         arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.5))

        # Resultant arrow
        ax2.annotate(f"Fa={Fa_r:.1f} kN/m",
                     xy=(-0.1, H - ha_r),
                     xytext=(Fa_r * scale * 0.5 + 0.5, H - ha_r),
                     arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=2.5),
                     fontsize=9, color="#1d4ed8", fontweight='bold')

        ax2.set_xlim(-0.5, 5)
        ax2.set_ylim(H + 0.5, -0.5)
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Depth (m)")
        ax2.set_title("Active Pressure (Rankine)", fontweight='bold', color='#1a3a5c')
        ax2.axhline(0, color='#475569', lw=1.5)
        ax2.axhline(H, color='#475569', lw=1.5)
        ax2.grid(True, alpha=0.25, linestyle='--')
        st.pyplot(fig2)
        plt.close()

# ════════════════════════════════════════════
# TAB 2 – Comparison Charts
# ════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Active Pressure – All Methods")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.plot(pa_rankine_net, depths, 'b-',  lw=2.2, label=f'Rankine (Ka={Ka_rankine:.3f})')
        ax3.plot(pa_coulomb_net, depths, 'g--', lw=2.2, label=f'Coulomb (Ka={Ka_coulomb:.3f})')
        ax3.plot(pa_atrest,      depths, 'm:',  lw=2.2, label=f'At-Rest (K0={K0_val:.3f})')
        ax3.invert_yaxis()
        ax3.set_xlabel("Active Pressure (kPa)")
        ax3.set_ylabel("Depth (m)")
        ax3.legend(); ax3.grid(True, alpha=0.3)
        ax3.set_title("Active Lateral Pressure Comparison")
        fig3.tight_layout()
        st.pyplot(fig3); plt.close()

    with col2:
        st.subheader("Passive Pressure – Rankine vs Coulomb")
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.plot(pp_rankine, depths, 'b-',  lw=2.2, label=f'Rankine (Kp={Kp_rankine:.3f})')
        ax4.plot(pp_coulomb, depths, 'g--', lw=2.2, label=f'Coulomb (Kp={Kp_coulomb:.3f})')
        ax4.invert_yaxis()
        ax4.set_xlabel("Passive Pressure (kPa)")
        ax4.set_ylabel("Depth (m)")
        ax4.legend(); ax4.grid(True, alpha=0.3)
        ax4.set_title("Passive Lateral Pressure Comparison")
        fig4.tight_layout()
        st.pyplot(fig4); plt.close()

    # Surcharge effect
    if surcharge_type != "None":
        st.subheader(f"Surcharge Effect – {surcharge_type}")
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        if surcharge_type == "Uniform":
            sur = Ka_rankine * q_uniform * np.ones_like(depths)
            ax5.plot(sur, depths, 'r-', lw=2, label=f'Uniform q={q_uniform} kPa')
        else:
            sur = surcharge_pressure(depths, surcharge_type,
                                     Q=Q_line or Q_point, x=x_line or x_point)
            ax5.plot(sur, depths, 'r-', lw=2, label=surcharge_type)
        ax5.invert_yaxis()
        ax5.set_xlabel("Surcharge-induced Pressure (kPa)")
        ax5.set_ylabel("Depth (m)")
        ax5.legend(); ax5.grid(True, alpha=0.3)
        ax5.set_title("Lateral Pressure from Surcharge (Boussinesq)")
        fig5.tight_layout()
        st.pyplot(fig5); plt.close()

    # Bar chart – resultant forces
    st.subheader("Resultant Force Comparison")
    fig6, axes6 = plt.subplots(1, 2, figsize=(10, 4))

    methods = ["Rankine", "Coulomb", "At-Rest"]
    fa_vals = [Fa_r, Fa_c, Fa_0]
    fp_vals = [Fp_r, Fp_c, 0]
    colors_a = ['#3b82f6', '#22c55e', '#a855f7']
    colors_p = ['#f59e0b', '#ef4444', '#d1d5db']

    bars = axes6[0].bar(methods, fa_vals, color=colors_a, edgecolor='white', linewidth=1.5)
    axes6[0].bar_label(bars, fmt='%.1f kN/m', padding=3, fontsize=9)
    axes6[0].set_title("Active Resultant Force (kN/m)")
    axes6[0].set_ylabel("Force (kN/m)")
    axes6[0].grid(axis='y', alpha=0.3)

    bars2 = axes6[1].bar(methods[:2], fp_vals[:2], color=colors_p[:2], edgecolor='white', linewidth=1.5)
    axes6[1].bar_label(bars2, fmt='%.1f kN/m', padding=3, fontsize=9)
    axes6[1].set_title("Passive Resultant Force (kN/m)")
    axes6[1].set_ylabel("Force (kN/m)")
    axes6[1].grid(axis='y', alpha=0.3)

    fig6.tight_layout()
    st.pyplot(fig6); plt.close()

# ════════════════════════════════════════════
# TAB 3 – Formulas & Theory
# ════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📘 Rankine Theory")
        st.markdown("""
**Assumptions:**
- No wall friction (smooth wall)
- Horizontal or uniformly sloping backfill
- Semi-infinite, homogeneous soil mass

**Active Coefficient (flat backfill):**
$$K_a = \\tan^2\\left(45° - \\frac{\\phi}{2}\\right) = \\frac{1 - \\sin\\phi}{1 + \\sin\\phi}$$

**Active Coefficient (sloped backfill β):**
$$K_a = \\cos\\beta \\cdot \\frac{\\cos\\beta - \\sqrt{\\cos^2\\beta - \\cos^2\\phi}}{\\cos\\beta + \\sqrt{\\cos^2\\beta - \\cos^2\\phi}}$$

**Passive Coefficient:**
$$K_p = \\tan^2\\left(45° + \\frac{\\phi}{2}\\right) = \\frac{1 + \\sin\\phi}{1 - \\sin\\phi}$$

**Lateral Pressure:**
$$\\sigma_a = K_a \\gamma z - 2c\\sqrt{K_a}$$
$$\\sigma_p = K_p \\gamma z + 2c\\sqrt{K_p}$$

**With uniform surcharge q:**
$$\\sigma_a = K_a(\\gamma z + q) - 2c\\sqrt{K_a}$$
        """)

        st.markdown("### 📗 At-Rest Pressure")
        st.markdown("""
**Jaky's formula (normally consolidated):**
$$K_0 = 1 - \\sin\\phi$$

**With backfill slope β:**
$$K_0 = (1 - \\sin\\phi)(1 + \\sin\\beta)$$

**Lateral Pressure:**
$$\\sigma_0 = K_0 \\gamma z + K_0 q$$

**Resultant:**
$$F_0 = \\frac{1}{2} K_0 \\gamma H^2 + K_0 q H$$

*Used for rigid structures with no lateral movement.*
        """)

    with c2:
        st.markdown("### 📙 Coulomb Theory")
        st.markdown("""
**Assumptions:**
- Wall friction δ considered
- Planar failure surface in soil
- More accurate for passive case

**Active Coefficient:**
$$K_a = \\frac{\\cos^2(\\phi - \\alpha)}{\\cos^2\\alpha \\cos(\\delta+\\alpha)\\left[1+\\sqrt{\\frac{\\sin(\\phi+\\delta)\\sin(\\phi-\\beta)}{\\cos(\\delta+\\alpha)\\cos(\\alpha-\\beta)}}\\right]^2}$$

**Passive Coefficient:**
$$K_p = \\frac{\\cos^2(\\phi + \\alpha)}{\\cos^2\\alpha \\cos(\\delta-\\alpha)\\left[1-\\sqrt{\\frac{\\sin(\\phi+\\delta)\\sin(\\phi+\\beta)}{\\cos(\\delta-\\alpha)\\cos(\\alpha-\\beta)}}\\right]^2}$$

Where:
- α = wall inclination from vertical
- β = backfill slope angle
- δ = wall–soil friction angle
        """)

        st.markdown("### 📕 Surcharge Effects")
        st.markdown("""
**Uniform Surcharge q (kPa):**
$$\\Delta\\sigma_h = K_a \\cdot q \\quad (\\text{constant with depth})$$

**Line Load Q (kN/m) at distance x (Boussinesq):**

For m = x/H > 0.4:
$$\\Delta\\sigma_h = \\frac{2Q}{\\pi H}\\cdot\\frac{m^2 n}{(m^2+n^2)^2}$$

For m ≤ 0.4:
$$\\Delta\\sigma_h = \\frac{0.203Q}{H}\\cdot\\frac{n}{(0.16+n^2)^2}$$

**Point Load Q (kN) (Boussinesq):**
$$\\Delta\\sigma_h = \\frac{3Q}{2\\pi H^2}\\cdot\\frac{m^2 n^3}{(m^2+n^2)^{2.5}}$$

where n = z/H, m = x/H
        """)

    st.markdown("---")
    st.markdown("### 🔎 Method Selection Guide")
    st.markdown("""
| Scenario | Recommended Method |
|---|---|
| Rigid wall, no movement (basement) | **At-Rest** |
| Cantilever / gravity wall (active) | **Rankine** (simple, conservative) |
| Wall with significant friction or batter | **Coulomb** |
| Passive resistance calculation | **Rankine** (safer, Coulomb overestimates) |
| Cohesive backfill | **Rankine** (includes 2c√Ka term) |
| Sloped backfill | Both methods support β ≠ 0 |
    """)

# ════════════════════════════════════════════
# TAB 4 – Detailed Tables
# ════════════════════════════════════════════
with tab4:
    st.subheader("Pressure Values at Selected Depths")

    import pandas as pd
    idx = np.linspace(0, n_layers - 1, min(20, n_layers), dtype=int)
    df = pd.DataFrame({
        "Depth (m)":        np.round(depths[idx], 2),
        "Rankine σa (kPa)": np.round(pa_rankine_net[idx], 2),
        "Coulomb σa (kPa)": np.round(pa_coulomb_net[idx], 2),
        "At-Rest σ0 (kPa)": np.round(pa_atrest[idx], 2),
        "Rankine σp (kPa)": np.round(pp_rankine[idx], 2),
        "Coulomb σp (kPa)": np.round(pp_coulomb[idx], 2),
    })
    st.dataframe(df, use_container_width=True)

    # Summary table
    st.subheader("Summary of Results")
    summary = pd.DataFrame({
        "Method":         ["Rankine", "Coulomb", "At-Rest"],
        "K (active)":     [f"{Ka_rankine:.4f}", f"{Ka_coulomb:.4f}", f"{K0_val:.4f}"],
        "K (passive)":    [f"{Kp_rankine:.4f}", f"{Kp_coulomb:.4f}", "—"],
        "Fa (kN/m)":      [f"{Fa_r:.2f}", f"{Fa_c:.2f}", f"{Fa_0:.2f}"],
        "ha from base (m)":[f"{ha_r:.2f}", f"{ha_c:.2f}", f"{ha_0:.2f}"],
        "Fp (kN/m)":      [f"{Fp_r:.2f}", f"{Fp_c:.2f}", "—"],
        "hp from base (m)":[f"{hp_r:.2f}", f"{hp_c:.2f}", "—"],
    })
    st.dataframe(summary, use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Pressure Table (CSV)", csv,
                           "pressure_data.csv", "text/csv")
    with col_dl2:
        csv2 = summary.to_csv(index=False)
        st.download_button("⬇️ Download Summary Table (CSV)", csv2,
                           "summary_results.csv", "text/csv")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#64748b;font-size:0.85rem;padding:10px'>
  🏗️ Lateral Earth Pressure Calculator &nbsp;|&nbsp;
  Rankine · Coulomb · At-Rest · Surcharge &nbsp;|&nbsp;
  Based on classical soil mechanics (Das, Coduto, Terzaghi)
</div>
""", unsafe_allow_html=True)
