import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Lateral Earth Pressure Calculator - English Units",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a3a5c 0%,#2d6a9f 100%);padding:20px 30px;border-radius:12px;margin-bottom:24px;color:white}.main-header h1{margin:0;font-size:2rem}.main-header p{margin:4px 0 0;opacity:.85;font-size:.95rem}.result-card{background:#f0f7ff;border-left:5px solid #2d6a9f;border-radius:8px;padding:16px 20px;margin:8px 0}.result-card h4{margin:0 0 8px;color:#1a3a5c}.method-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.78rem;font-weight:600;margin-bottom:6px}.rankine{background:#dbeafe;color:#1e40af}.coulomb{background:#dcfce7;color:#166534}.atrest{background:#fef9c3;color:#854d0e}.note{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:10px 14px;color:#7c2d12}.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{background:#f1f5f9;border-radius:8px 8px 0 0;padding:8px 20px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🏗️ Lateral Earth Pressure Calculator</h1>
  <p>English Units · Multi-Layer Soil · Water Table · Rankine · Coulomb · At-Rest · Surcharge</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Soil layer table defaults
# -----------------------------
def default_layers(n, height):
    bottoms = np.linspace(height / n, height, n)
    return pd.DataFrame({
        "Bottom Depth (ft)": np.round(bottoms, 2),
        "Moist Unit Weight γm (pcf)": [120.0] * n,
        "Saturated Unit Weight γsat (pcf)": [125.0] * n,
        "Friction Angle φ (deg)": [30.0] * n,
        "Cohesion c (psf)": [0.0] * n,
    })

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("⚙️ Input Parameters")

    st.subheader("🧱 Wall Geometry")
    H = st.number_input("Wall Height H (ft)", 3.0, 100.0, 20.0, 1.0)
    alpha_deg = st.number_input("Wall Inclination α (°)", -20.0, 20.0, 0.0, 1.0,
                                help="Angle of wall face from vertical (+ = leaning into backfill)")
    beta_deg = st.number_input("Backfill Slope β (°)", 0.0, 30.0, 0.0, 1.0)
    delta_deg = st.number_input("Wall Friction δ (°), for Coulomb", 0.0, 35.0, 15.0, 1.0)

    st.subheader("💧 Groundwater")
    include_water = st.checkbox("Include water table", True)
    water_table = st.number_input("Water Table Depth from Top (ft)", 0.0, float(H), min(10.0, float(H)), 0.5)
    gamma_w = st.number_input("Unit Weight of Water γw (pcf)", 60.0, 64.0, 62.4, 0.1)

    st.subheader("🪨 Soil Properties / Layers")
    st.caption("Use the Add Layer button to create more layers. Depths are measured downward from the retained ground surface.")

    if "soil_layers" not in st.session_state:
        st.session_state.soil_layers = default_layers(1, float(H)).to_dict("records")

    if st.button("➕ Add Layer", use_container_width=True):
        layers = st.session_state.soil_layers
        if len(layers) < 8:
            if layers:
                previous_top = 0.0 if len(layers) == 1 else float(layers[-2]["Bottom Depth (ft)"])
                split_depth = round((previous_top + float(H)) / 2.0, 2)
                layers[-1]["Bottom Depth (ft)"] = max(previous_top + 0.5, min(split_depth, float(H) - 0.5))
                template = layers[-1].copy()
                template["Bottom Depth (ft)"] = float(H)
            else:
                template = default_layers(1, float(H)).iloc[0].to_dict()
            layers.append(template)
            st.session_state.soil_layers = layers
        else:
            st.warning("Maximum 8 layers allowed.")

    if len(st.session_state.soil_layers) > 1:
        if st.button("➖ Remove Last Layer", use_container_width=True):
            st.session_state.soil_layers = st.session_state.soil_layers[:-1]
            st.session_state.soil_layers[-1]["Bottom Depth (ft)"] = float(H)

    updated_layers = []
    previous_bottom = 0.0
    for i, layer in enumerate(st.session_state.soil_layers):
        with st.container(border=True):
            st.markdown(f"**Layer {i + 1}**")
            bottom_default = min(float(layer.get("Bottom Depth (ft)", H)), float(H))
            if i == len(st.session_state.soil_layers) - 1:
                bottom_min = min(float(H), previous_bottom + 0.1)
                bottom_value = st.number_input(
                    "Bottom Depth from Top (ft)",
                    min_value=bottom_min,
                    max_value=float(H),
                    value=float(H),
                    step=0.5,
                    key=f"layer_{i}_bottom",
                    help="Last layer bottom is set to the wall height."
                )
            else:
                bottom_min = min(float(H), previous_bottom + 0.1)
                bottom_value = st.number_input(
                    "Bottom Depth from Top (ft)",
                    min_value=bottom_min,
                    max_value=float(H),
                    value=max(bottom_min, bottom_default),
                    step=0.5,
                    key=f"layer_{i}_bottom",
                )

            gamma_m_value = st.number_input(
                "Moist Unit Weight γm (pcf)",
                min_value=50.0,
                max_value=160.0,
                value=float(layer.get("Moist Unit Weight γm (pcf)", 120.0)),
                step=1.0,
                key=f"layer_{i}_gamma_m",
            )
            gamma_sat_value = st.number_input(
                "Saturated Unit Weight γsat (pcf)",
                min_value=50.0,
                max_value=170.0,
                value=float(layer.get("Saturated Unit Weight γsat (pcf)", 125.0)),
                step=1.0,
                key=f"layer_{i}_gamma_sat",
            )
            phi_value = st.number_input(
                "Friction Angle φ (deg)",
                min_value=0.0,
                max_value=50.0,
                value=float(layer.get("Friction Angle φ (deg)", 30.0)),
                step=1.0,
                key=f"layer_{i}_phi",
            )
            c_value = st.number_input(
                "Cohesion c (psf)",
                min_value=0.0,
                max_value=5000.0,
                value=float(layer.get("Cohesion c (psf)", 0.0)),
                step=50.0,
                key=f"layer_{i}_c",
            )

        previous_bottom = float(bottom_value)
        updated_layers.append({
            "Bottom Depth (ft)": float(bottom_value),
            "Moist Unit Weight γm (pcf)": float(gamma_m_value),
            "Saturated Unit Weight γsat (pcf)": float(gamma_sat_value),
            "Friction Angle φ (deg)": float(phi_value),
            "Cohesion c (psf)": float(c_value),
        })

    st.session_state.soil_layers = updated_layers
    layer_df = pd.DataFrame(updated_layers)

    st.subheader("📦 Surcharge")
    surcharge_type = st.selectbox("Surcharge Type", ["None", "Uniform", "Line Load", "Point Load"])
    q_uniform = 0.0
    Q_line = 0.0
    x_line = 5.0
    Q_point = 0.0
    x_point = 5.0

    if surcharge_type == "Uniform":
        q_uniform = st.number_input("Uniform Surcharge q (psf)", 0.0, 10000.0, 250.0, 25.0)
    elif surcharge_type == "Line Load":
        Q_line = st.number_input("Line Load Q (lb/ft)", 0.0, 50000.0, 2000.0, 100.0)
        x_line = st.number_input("Distance from Wall x (ft)", 0.5, 100.0, 6.0, 0.5)
    elif surcharge_type == "Point Load":
        Q_point = st.number_input("Point Load Q (lb)", 0.0, 200000.0, 10000.0, 500.0)
        x_point = st.number_input("Distance from Wall x (ft)", 0.5, 100.0, 6.0, 0.5)

    st.subheader("📊 Display Options")
    n_points = st.slider("Pressure Diagram Points", 50, 500, 151)
    show_cohesion = st.checkbox("Include cohesion term", True)
    show_total_pressure = st.checkbox("Show total pressure including water", True)

# -----------------------------
# Soil layer cleanup
# -----------------------------
layer_df = layer_df.sort_values("Bottom Depth (ft)").reset_index(drop=True)
if layer_df.iloc[-1]["Bottom Depth (ft)"] < H:
    st.warning("The last soil layer bottom is less than the wall height. The app will extend the last layer to the wall base.")
    layer_df.loc[layer_df.index[-1], "Bottom Depth (ft)"] = H
elif layer_df.iloc[-1]["Bottom Depth (ft)"] > H:
    layer_df.loc[layer_df.index[-1], "Bottom Depth (ft)"] = H

# -----------------------------
# Calculations
# -----------------------------
alpha = np.radians(alpha_deg)
beta = np.radians(beta_deg)
delta = np.radians(delta_deg)
depths = np.linspace(0.0, H, int(n_points))


def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))


def rankine_Ka(phi, beta):
    if abs(beta) < 1e-9:
        return np.tan(np.pi / 4 - phi / 2) ** 2
    rad = np.cos(beta) ** 2 - np.cos(phi) ** 2
    if rad < 0:
        return np.nan
    num = np.cos(beta) - np.sqrt(rad)
    den = np.cos(beta) + np.sqrt(rad)
    return np.cos(beta) * num / den


def rankine_Kp(phi, beta):
    if abs(beta) < 1e-9:
        return np.tan(np.pi / 4 + phi / 2) ** 2
    rad = np.cos(beta) ** 2 - np.cos(phi) ** 2
    if rad < 0:
        return np.nan
    num = np.cos(beta) + np.sqrt(rad)
    den = np.cos(beta) - np.sqrt(rad)
    return np.cos(beta) * num / den


def coulomb_Ka(phi, delta, alpha, beta):
    try:
        rad = (np.sin(phi + delta) * np.sin(phi - beta)) / (np.cos(delta + alpha) * np.cos(alpha - beta))
        if rad < 0:
            return np.nan
        num = np.cos(phi - alpha) ** 2
        den = np.cos(alpha) ** 2 * np.cos(delta + alpha) * (1 + np.sqrt(rad)) ** 2
        return num / den
    except Exception:
        return np.nan


def coulomb_Kp(phi, delta, alpha, beta):
    try:
        rad = (np.sin(phi + delta) * np.sin(phi + beta)) / (np.cos(delta - alpha) * np.cos(alpha - beta))
        if rad < 0:
            return np.nan
        num = np.cos(phi + alpha) ** 2
        den = np.cos(alpha) ** 2 * np.cos(delta - alpha) * (1 - np.sqrt(rad)) ** 2
        if abs(den) < 1e-12:
            return np.nan
        return num / den
    except Exception:
        return np.nan


def K0(phi, beta):
    return (1 - np.sin(phi)) * (1 + np.sin(beta))


def layer_index_at_depth(z):
    bottoms = layer_df["Bottom Depth (ft)"].to_numpy(dtype=float)
    return min(int(np.searchsorted(bottoms, z, side="left")), len(bottoms) - 1)


def properties_at_depth(z):
    row = layer_df.iloc[layer_index_at_depth(z)]
    phi = np.radians(float(row["Friction Angle φ (deg)"]))
    c = float(row["Cohesion c (psf)"])
    return phi, c


def vertical_stresses_at_depth(z):
    """Returns total vertical stress, pore pressure, and effective vertical stress at depth z."""
    if z <= 0:
        return 0.0, 0.0, 0.0

    total = 0.0
    top = 0.0
    for _, row in layer_df.iterrows():
        bottom = min(float(row["Bottom Depth (ft)"]), H)
        if z <= top:
            break
        seg_top = top
        seg_bot = min(z, bottom)
        if seg_bot > seg_top:
            gm = float(row["Moist Unit Weight γm (pcf)"])
            gsat = float(row["Saturated Unit Weight γsat (pcf)"])
            if include_water:
                if seg_bot <= water_table:
                    total += gm * (seg_bot - seg_top)
                elif seg_top >= water_table:
                    total += gsat * (seg_bot - seg_top)
                else:
                    total += gm * (water_table - seg_top) + gsat * (seg_bot - water_table)
            else:
                total += gm * (seg_bot - seg_top)
        top = bottom
        if top >= z:
            break

    u = gamma_w * max(0.0, z - water_table) if include_water else 0.0
    effective = max(total - u, 0.0)
    return total, u, effective


def surcharge_pressure(z_arr, stype, q=0, Q=0, x=1, K_arr=None):
    p = np.zeros_like(z_arr, dtype=float)
    if K_arr is None:
        K_arr = np.ones_like(z_arr, dtype=float)
    if stype == "Uniform":
        p = K_arr * q
    elif stype == "Line Load":
        for i, z in enumerate(z_arr):
            if z <= 0:
                continue
            m = x / H
            n = z / H
            if m > 0.4:
                p[i] = (2 * Q / np.pi / H) * (m ** 2 * n / (m ** 2 + n ** 2) ** 2)
            else:
                p[i] = (0.203 * Q / H) * (n / (0.16 + n ** 2) ** 2)
    elif stype == "Point Load":
        for i, z in enumerate(z_arr):
            if z <= 0:
                continue
            m = x / H
            n = z / H
            if m > 0.4:
                p[i] = (3 * Q / (2 * np.pi * H ** 2)) * (m ** 2 * n ** 3 / (m ** 2 + n ** 2) ** 2.5)
            else:
                p[i] = (0.28 * Q / H ** 2) * (n ** 3 / (0.16 + n ** 2) ** 3)
    return p

sigma_v_total = np.zeros_like(depths)
u_water = np.zeros_like(depths)
sigma_v_eff = np.zeros_like(depths)
phi_arr = np.zeros_like(depths)
c_arr = np.zeros_like(depths)
Ka_r = np.zeros_like(depths)
Kp_r = np.zeros_like(depths)
Ka_c = np.zeros_like(depths)
Kp_c = np.zeros_like(depths)
K0_arr = np.zeros_like(depths)

for i, z in enumerate(depths):
    sigma_v_total[i], u_water[i], sigma_v_eff[i] = vertical_stresses_at_depth(z)
    phi_i, c_i = properties_at_depth(z)
    phi_arr[i] = phi_i
    c_arr[i] = c_i
    Ka_r[i] = rankine_Ka(phi_i, beta)
    Kp_r[i] = rankine_Kp(phi_i, beta)
    Ka_c[i] = coulomb_Ka(phi_i, delta, alpha, beta)
    Kp_c[i] = coulomb_Kp(phi_i, delta, alpha, beta)
    K0_arr[i] = K0(phi_i, beta)

# Replace invalid Coulomb values with Rankine fallback so the app does not crash.
Ka_c = np.where(np.isfinite(Ka_c), Ka_c, Ka_r)
Kp_c = np.where(np.isfinite(Kp_c), Kp_c, Kp_r)

q_for_uniform = q_uniform if surcharge_type == "Uniform" else 0.0
sur_r = surcharge_pressure(depths, surcharge_type, q=q_uniform, Q=Q_line if surcharge_type == "Line Load" else Q_point,
                           x=x_line if surcharge_type == "Line Load" else x_point, K_arr=Ka_r)
sur_c = surcharge_pressure(depths, surcharge_type, q=q_uniform, Q=Q_line if surcharge_type == "Line Load" else Q_point,
                           x=x_line if surcharge_type == "Line Load" else x_point, K_arr=Ka_c)
sur_0 = surcharge_pressure(depths, surcharge_type, q=q_uniform, Q=Q_line if surcharge_type == "Line Load" else Q_point,
                           x=x_line if surcharge_type == "Line Load" else x_point, K_arr=K0_arr)

cohesion_term_r = 2 * c_arr * safe_sqrt(Ka_r) if show_cohesion else 0.0
cohesion_term_c = 2 * c_arr * safe_sqrt(Ka_c) if show_cohesion else 0.0

pa_rankine_eff = Ka_r * sigma_v_eff - cohesion_term_r + sur_r
pa_coulomb_eff = Ka_c * sigma_v_eff - cohesion_term_c + sur_c
pa_atrest_eff = K0_arr * sigma_v_eff + sur_0

pa_rankine_eff_net = np.clip(pa_rankine_eff, 0, None)
pa_coulomb_eff_net = np.clip(pa_coulomb_eff, 0, None)
pa_atrest_eff_net = np.clip(pa_atrest_eff, 0, None)

pp_rankine_eff = Kp_r * sigma_v_eff + 2 * c_arr * safe_sqrt(Kp_r) + (Kp_r * q_uniform if surcharge_type == "Uniform" else 0)
pp_coulomb_eff = Kp_c * sigma_v_eff + 2 * c_arr * safe_sqrt(Kp_c) + (Kp_c * q_uniform if surcharge_type == "Uniform" else 0)

water_component = u_water if show_total_pressure else 0.0
pa_rankine = pa_rankine_eff_net + water_component
pa_coulomb = pa_coulomb_eff_net + water_component
pa_atrest = pa_atrest_eff_net + water_component
pp_rankine = pp_rankine_eff + water_component
pp_coulomb = pp_coulomb_eff + water_component


def resultant(p, z):
    F_lb_per_ft = float(np.trapezoid(p, z))
    if F_lb_per_ft <= 0:
        return 0.0, H / 3
    moment_about_base = float(np.trapezoid(p * (H - z), z))
    h_bar = moment_about_base / F_lb_per_ft
    return F_lb_per_ft / 1000.0, h_bar

Fa_r, ha_r = resultant(pa_rankine, depths)
Fa_c, ha_c = resultant(pa_coulomb, depths)
Fa_0, ha_0 = resultant(pa_atrest, depths)
Fp_r, hp_r = resultant(pp_rankine, depths)
Fp_c, hp_c = resultant(pp_coulomb, depths)

# -----------------------------
# Results UI
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📐 Results & Diagrams", "📊 Comparison Charts", "📋 Formulas & Notes", "🔢 Detailed Tables"])

with tab1:
    col_res, col_diag = st.columns([1, 1.6])
    with col_res:
        st.subheader("Layer Summary")
        display_layers = layer_df.copy()
        display_layers.insert(0, "Top Depth (ft)", [0.0] + list(display_layers["Bottom Depth (ft)"].iloc[:-1]))
        st.dataframe(display_layers, use_container_width=True, hide_index=True)

        st.subheader("Active Pressure Resultants")
        def result_card(method, cls, F, h):
            st.markdown(f"""
            <div class="result-card">
              <span class="method-badge {cls}">{method}</span>
              <h4>F<sub>a</sub> = {F:.2f} kips/ft</h4>
              <p style="margin:0;font-size:0.85rem;">Acts at <b>{h:.2f} ft</b> above base</p>
            </div>""", unsafe_allow_html=True)

        result_card("Rankine", "rankine", Fa_r, ha_r)
        result_card("Coulomb", "coulomb", Fa_c, ha_c)
        result_card("At-Rest", "atrest", Fa_0, ha_0)

        st.subheader("Passive Pressure Resultants")
        st.markdown(f"""
        <div class="result-card"><span class="method-badge rankine">Rankine</span><h4>F<sub>p</sub> = {Fp_r:.2f} kips/ft</h4><p style="margin:0;font-size:0.85rem;">Acts at <b>{hp_r:.2f} ft</b> above base</p></div>
        <div class="result-card"><span class="method-badge coulomb">Coulomb</span><h4>F<sub>p</sub> = {Fp_c:.2f} kips/ft</h4><p style="margin:0;font-size:0.85rem;">Acts at <b>{hp_c:.2f} ft</b> above base</p></div>
        """, unsafe_allow_html=True)

        if include_water:
            st.markdown(f"<div class='note'>Water pressure is added below <b>{water_table:.2f} ft</b> when total pressure is shown.</div>", unsafe_allow_html=True)

    with col_diag:
        fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
        fig.patch.set_facecolor("#f8fafc")
        configs = [
            ("Rankine", pa_rankine, pp_rankine, "#3b82f6", "#f59e0b"),
            ("Coulomb", pa_coulomb, pp_coulomb, "#22c55e", "#ef4444"),
            ("At-Rest", pa_atrest, pa_atrest, "#a855f7", "#a855f7"),
        ]
        for ax, (title, pa, pp, c_a, c_p) in zip(axes, configs):
            ax.set_facecolor("#f8fafc")
            ax.fill_betweenx(depths, 0, pp, alpha=0.15, color=c_p)
            ax.plot(pp, depths, color=c_p, lw=2, label="Passive" if title != "At-Rest" else "At-Rest")
            ax.fill_betweenx(depths, 0, -pa, alpha=0.20, color=c_a)
            ax.plot(-pa, depths, color=c_a, lw=2.2, label="Active")
            for b in layer_df["Bottom Depth (ft)"].iloc[:-1]:
                ax.axhline(b, color="#64748b", lw=0.8, ls="--", alpha=0.6)
            if include_water:
                ax.axhline(water_table, color="#0ea5e9", lw=1.2, ls=":", alpha=0.9)
            ax.axvline(0, color="#334155", lw=0.8)
            ax.invert_yaxis()
            ax.set_title(title, fontweight="bold", fontsize=12, color="#1a3a5c")
            ax.set_xlabel("Pressure (psf)", fontsize=9)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=8)
        axes[0].set_ylabel("Depth (ft)", fontsize=10)
        fig.suptitle("Lateral Earth Pressure Diagrams", fontsize=13, fontweight="bold", color="#1a3a5c", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active / At-Rest Pressure")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.plot(pa_rankine, depths, lw=2.2, label="Rankine Active")
        ax3.plot(pa_coulomb, depths, lw=2.2, ls="--", label="Coulomb Active")
        ax3.plot(pa_atrest, depths, lw=2.2, ls=":", label="At-Rest")
        ax3.invert_yaxis(); ax3.set_xlabel("Pressure (psf)"); ax3.set_ylabel("Depth (ft)")
        ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_title("Lateral Pressure Comparison")
        fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)
    with col2:
        st.subheader("Passive Pressure")
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.plot(pp_rankine, depths, lw=2.2, label="Rankine Passive")
        ax4.plot(pp_coulomb, depths, lw=2.2, ls="--", label="Coulomb Passive")
        ax4.invert_yaxis(); ax4.set_xlabel("Pressure (psf)"); ax4.set_ylabel("Depth (ft)")
        ax4.legend(); ax4.grid(True, alpha=0.3); ax4.set_title("Passive Pressure Comparison")
        fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    st.subheader("Resultant Force Comparison")
    fig6, axes6 = plt.subplots(1, 2, figsize=(10, 4))
    methods = ["Rankine", "Coulomb", "At-Rest"]
    bars = axes6[0].bar(methods, [Fa_r, Fa_c, Fa_0])
    axes6[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes6[0].set_title("Active / At-Rest Resultant (kips/ft)"); axes6[0].set_ylabel("kips/ft"); axes6[0].grid(axis="y", alpha=0.3)
    bars2 = axes6[1].bar(methods[:2], [Fp_r, Fp_c])
    axes6[1].bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    axes6[1].set_title("Passive Resultant (kips/ft)"); axes6[1].set_ylabel("kips/ft"); axes6[1].grid(axis="y", alpha=0.3)
    fig6.tight_layout(); st.pyplot(fig6); plt.close(fig6)

with tab3:
    st.markdown("""
### How this version handles layers and groundwater

- The app computes vertical stress by integrating each soil layer from the surface to the selected depth.
- Above the water table it uses moist unit weight, γm.
- Below the water table it uses saturated unit weight, γsat, computes pore pressure, then uses effective vertical stress for soil pressure.
- Total lateral pressure = effective lateral soil pressure + water pressure.
- If cohesion creates negative active pressure, the active pressure is clipped to zero for the net diagram.

### English-unit conventions

| Quantity | Unit |
|---|---|
| Depth / wall height | ft |
| Soil unit weight | pcf |
| Cohesion | psf |
| Pressure | psf |
| Uniform surcharge | psf |
| Line load | lb/ft |
| Point load | lb |
| Resultant force | kips/ft |

### Key formulas

Rankine active pressure, effective-stress form:

$$p'_a = K_a \sigma'_v - 2c\sqrt{K_a}$$

Total pressure below groundwater:

$$p_{total} = p' + u$$

Hydrostatic pore pressure:

$$u = \gamma_w(z-z_w)$$

where $z_w$ is the water table depth.
""")

with tab4:
    st.subheader("Pressure Values at Selected Depths")
    idx = np.linspace(0, len(depths) - 1, min(30, len(depths)), dtype=int)
    df = pd.DataFrame({
        "Depth (ft)": np.round(depths[idx], 2),
        "Layer": [layer_index_at_depth(float(z)) + 1 for z in depths[idx]],
        "σv total (psf)": np.round(sigma_v_total[idx], 1),
        "u water (psf)": np.round(u_water[idx], 1),
        "σv effective (psf)": np.round(sigma_v_eff[idx], 1),
        "Ka Rankine": np.round(Ka_r[idx], 3),
        "Ka Coulomb": np.round(Ka_c[idx], 3),
        "Rankine active total (psf)": np.round(pa_rankine[idx], 1),
        "Coulomb active total (psf)": np.round(pa_coulomb[idx], 1),
        "At-rest total (psf)": np.round(pa_atrest[idx], 1),
        "Rankine passive total (psf)": np.round(pp_rankine[idx], 1),
        "Coulomb passive total (psf)": np.round(pp_coulomb[idx], 1),
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Summary of Results")
    summary = pd.DataFrame({
        "Method": ["Rankine", "Coulomb", "At-Rest"],
        "Fa / F0 (kips/ft)": [f"{Fa_r:.2f}", f"{Fa_c:.2f}", f"{Fa_0:.2f}"],
        "Height above base (ft)": [f"{ha_r:.2f}", f"{ha_c:.2f}", f"{ha_0:.2f}"],
        "Fp (kips/ft)": [f"{Fp_r:.2f}", f"{Fp_c:.2f}", "—"],
        "Passive height above base (ft)": [f"{hp_r:.2f}", f"{hp_c:.2f}", "—"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download Pressure Table (CSV)", df.to_csv(index=False), "pressure_data_english.csv", "text/csv")
    with col_dl2:
        st.download_button("⬇️ Download Summary Table (CSV)", summary.to_csv(index=False), "summary_results_english.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#64748b;font-size:0.85rem;padding:10px'>
  🏗️ Lateral Earth Pressure Calculator | English Units | Multi-Layer Soil | Groundwater | Rankine · Coulomb · At-Rest
</div>
""", unsafe_allow_html=True)
