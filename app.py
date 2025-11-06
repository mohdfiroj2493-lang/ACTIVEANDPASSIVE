import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# ----------------------------
# CONSTANTS
# ----------------------------
GAMMA_W = 62.4  # pcf (unit weight of water, lb/ft³)

# ----------------------------
# DATA STRUCTURES
# ----------------------------
@dataclass
class SoilLayer:
    thickness: float      # ft
    phi_deg: float        # degrees
    cohesion: float       # psf
    gamma_dry: float      # pcf
    gamma_sat: float      # pcf

@dataclass
class Profile:
    layers: List[SoilLayer] = field(default_factory=list)
    water_table_depth: float = 0.0
    dz: float = 0.1  # ft

    def total_depth(self):
        return sum(L.thickness for L in self.layers)

    def layer_at_depth(self, z):
        s = 0.0
        for i, L in enumerate(self.layers):
            s += L.thickness
            if z <= s + 1e-6:
                return i
        return len(self.layers) - 1

    def ka_kp(self, phi_deg):
        sinp = math.sin(math.radians(phi_deg))
        ka = (1 - sinp) / (1 + sinp)
        kp = (1 + sinp) / (1 - sinp)
        return ka, kp

    def effective_gamma(self, L: SoilLayer):
        return L.gamma_sat - GAMMA_W

    def compute(self):
        H = self.total_depth()
        z = np.arange(0, H + self.dz, self.dz)
        sigma_v_eff = np.zeros_like(z)
        u = np.zeros_like(z)

        running = 0.0
        for k in range(1, len(z)):
            dz_loc = z[k] - z[k-1]
            z_mid = 0.5 * (z[k] + z[k-1])
            i = self.layer_at_depth(z_mid)
            L = self.layers[i]

            if z_mid < self.water_table_depth:
                gamma_eff = L.gamma_dry
            else:
                gamma_eff = self.effective_gamma(L)

            running += gamma_eff * dz_loc
            sigma_v_eff[k] = running
            u[k] = GAMMA_W * max(0.0, z[k] - self.water_table_depth)

        sigma_h_a = np.zeros_like(z)
        sigma_h_p = np.zeros_like(z)

        for k in range(len(z)):
            L = self.layers[self.layer_at_depth(z[k])]
            ka, kp = self.ka_kp(L.phi_deg)
            c = L.cohesion
            s_a = ka * sigma_v_eff[k] - 2 * c * math.sqrt(ka)
            s_p = kp * sigma_v_eff[k] + 2 * c * math.sqrt(kp)
            sigma_h_a[k] = max(0, s_a + u[k])
            sigma_h_p[k] = max(0, s_p + u[k])

        return {"z": z, "active": sigma_h_a, "passive": sigma_h_p}

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="Earth Pressure (English Units)", layout="wide")
st.title("🧱 Active & Passive Earth Pressure (English Units)")

st.sidebar.header("Soil and Water Parameters")

num_layers = st.sidebar.number_input("Number of soil layers", 1, 10, 3)
wt_active = st.sidebar.number_input("Water Table Depth (Active Side, ft)", 0.0, 50.0, 5.0, step=0.5)
wt_passive = st.sidebar.number_input("Water Table Depth (Passive Side, ft)", 0.0, 50.0, 8.0, step=0.5)
excavation_depth = st.sidebar.number_input("Excavation Depth (Passive Side, ft)", 0.0, 50.0, 3.0, step=0.5)

active_layers, passive_layers = [], []

st.sidebar.markdown("### Active Side Layers")
for i in range(int(num_layers)):
    st.sidebar.markdown(f"**Layer {i+1}**")
    t = st.sidebar.number_input(f"Thickness L{i+1} (ft)", 0.1, 50.0, 3.0)
    phi = st.sidebar.number_input(f"φ L{i+1} (°)", 0.0, 50.0, 30.0)
    c = st.sidebar.number_input(f"c L{i+1} (psf)", 0.0, 2000.0, 0.0)
    gd = st.sidebar.number_input(f"γ_dry L{i+1} (pcf)", 60.0, 140.0, 110.0)
    gs = st.sidebar.number_input(f"γ_sat L{i+1} (pcf)", 60.0, 140.0, 120.0)
    active_layers.append(SoilLayer(t, phi, c, gd, gs))
    passive_layers.append(SoilLayer(t, phi, c, gd, gs))  # for now identical

if st.sidebar.button("Compute"):
    prof_active = Profile(layers=active_layers, water_table_depth=wt_active)
    prof_passive = Profile(layers=passive_layers, water_table_depth=wt_passive)
    resA = prof_active.compute()
    resP = prof_passive.compute()

    z = resA["z"]
    fig, ax = plt.subplots(figsize=(7, 8))

    # Plot pressures on both sides
    ax.plot(-resA["active"], z, color="blue", label="Active Pressure (left)")
    ax.plot(resP["passive"], z, color="red", label="Passive Pressure (right)")

    # Draw wall and excavation line
    ax.axvline(0, color="black", linewidth=3)
    ax.fill_betweenx(z, -resA["active"], 0, color="lightblue", alpha=0.3)
    ax.fill_betweenx(z, 0, resP["passive"], color="salmon", alpha=0.3)

    # Show excavation line
    ax.plot([0, 5], [excavation_depth, excavation_depth], "k--", linewidth=1)

    ax.invert_yaxis()
    ax.set_xlabel("Lateral Pressure (psf)")
    ax.set_ylabel("Depth (ft)")
    ax.set_title("Active & Passive Earth Pressure Distribution")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("### Model Geometry Overview")
    st.markdown(f"- Total height = **{prof_active.total_depth():.2f} ft**")
    st.markdown(f"- Water table (active) = **{wt_active:.2f} ft**, (passive) = **{wt_passive:.2f} ft**")
    st.markdown(f"- Excavation depth = **{excavation_depth:.2f} ft**")
    st.success("✅ Computation complete and visualized.")
