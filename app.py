import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple

# ----------------------------
# CONSTANTS
# ----------------------------
GAMMA_W = 9.81  # kN/m³ (unit weight of water)

# ----------------------------
# SOIL LAYER DATA STRUCTURE
# ----------------------------
@dataclass
class SoilLayer:
    thickness: float                 # m
    phi_deg: float                   # degrees
    cohesion: float                  # kPa
    gamma_dry: float                 # kN/m³
    gamma_sat: float                 # kN/m³

    def __post_init__(self):
        assert self.thickness > 0, "Layer thickness must be positive."
        assert 0 <= self.phi_deg <= 50, "φ should be between 0° and 50°."
        assert self.gamma_sat >= self.gamma_dry, "γ_sat should be ≥ γ_dry."
        assert self.gamma_dry > 0 and self.gamma_sat > 0, "γ values must be positive."
        assert self.cohesion >= 0, "Cohesion cannot be negative."

# ----------------------------
# PROFILE CLASS
# ----------------------------
@dataclass
class Profile:
    layers: List[SoilLayer] = field(default_factory=list)
    water_table_depth: float = 0.0
    dz: float = 0.05

    def total_depth(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    def _layer_at_depth(self, z: float) -> int:
        s = 0.0
        for i, L in enumerate(self.layers):
            s_next = s + L.thickness
            if z <= s_next + 1e-9:
                return i
            s = s_next
        return len(self.layers) - 1

    def ka_kp(self, phi_deg: float) -> Tuple[float, float]:
        sinp = math.sin(math.radians(phi_deg))
        ka = (1 - sinp) / (1 + sinp)
        kp = (1 + sinp) / (1 - sinp) if sinp < 1 else float("inf")
        return ka, kp

    def effective_unit_weight(self, gamma_sat: float) -> float:
        return gamma_sat - GAMMA_W

    def compute_profiles(self):
        H = self.total_depth()
        z = np.arange(0, H + self.dz, self.dz)

        sigma_v_eff = np.zeros_like(z)
        running = 0.0
        for k in range(1, len(z)):
            dz_loc = z[k] - z[k-1]
            z_mid = 0.5 * (z[k] + z[k-1])
            li = self._layer_at_depth(z_mid)
            L = self.layers[li]

            if z_mid < self.water_table_depth - 1e-12:
                gamma_eff = L.gamma_dry
            else:
                gamma_eff = self.effective_unit_weight(L.gamma_sat)

            running += gamma_eff * dz_loc
            sigma_v_eff[k] = running

        u = GAMMA_W * np.clip(z - self.water_table_depth, a_min=0.0, a_max=None)

        sigma_h_a_eff = np.zeros_like(z)
        sigma_h_p_eff = np.zeros_like(z)

        for k in range(len(z)):
            z_ref = z[k] if k == 0 else 0.5 * (z[k] + z[k - 1])
            li = self._layer_at_depth(z_ref)
            L = self.layers[li]
            ka, kp = self.ka_kp(L.phi_deg)
            c = L.cohesion

            s_a = ka * sigma_v_eff[k] - 2.0 * c * math.sqrt(ka)
            s_p = kp * sigma_v_eff[k] + 2.0 * c * math.sqrt(kp)

            sigma_h_a_eff[k] = max(0.0, s_a)
            sigma_h_p_eff[k] = max(0.0, s_p)

        sigma_h_a_tot = sigma_h_a_eff + u
        sigma_h_p_tot = sigma_h_p_eff + u

        return {
            "z": z,
            "sigma_v_eff": sigma_v_eff,
            "u": u,
            "sigma_h_a_eff": sigma_h_a_eff,
            "sigma_h_p_eff": sigma_h_p_eff,
            "sigma_h_a_tot": sigma_h_a_tot,
            "sigma_h_p_tot": sigma_h_p_tot,
        }

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="Earth Pressure Calculator", layout="wide")

st.title("🌍 Active & Passive Earth Pressure Calculator")
st.markdown("### Rankine Theory (with Cohesion and Water Table)")

st.sidebar.header("Input Parameters")

num_layers = st.sidebar.number_input("Number of Soil Layers", 1, 10, 3)
water_table_depth = st.sidebar.number_input("Water Table Depth (m)", 0.0, 50.0, 1.5, step=0.1)

layers = []
for i in range(int(num_layers)):
    st.sidebar.subheader(f"Layer {i+1}")
    thickness = st.sidebar.number_input(f"Thickness L{i+1} (m)", 0.1, 50.0, 2.0)
    phi = st.sidebar.number_input(f"φ L{i+1} (°)", 0.0, 50.0, 30.0)
    cohesion = st.sidebar.number_input(f"c L{i+1} (kPa)", 0.0, 200.0, 0.0)
    gamma_dry = st.sidebar.number_input(f"γ_dry L{i+1} (kN/m³)", 5.0, 30.0, 17.5)
    gamma_sat = st.sidebar.number_input(f"γ_sat L{i+1} (kN/m³)", 5.0, 30.0, 19.5)
    layers.append(SoilLayer(thickness, phi, cohesion, gamma_dry, gamma_sat))

if st.sidebar.button("Compute"):
    try:
        prof = Profile(layers=layers, water_table_depth=water_table_depth)
        res = prof.compute_profiles()

        st.success("✅ Computation Complete!")
        st.write(f"**Total depth:** {prof.total_depth():.2f} m")

        # Plot active and passive pressures
        fig, ax = plt.subplots()
        ax.plot(res["sigma_h_a_tot"], res["z"], label="Active (Total)")
        ax.plot(res["sigma_h_p_tot"], res["z"], label="Passive (Total)")
        ax.invert_yaxis()
        ax.set_xlabel("Lateral Pressure (kPa)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Show data table
        st.subheader("Detailed Results")
        st.dataframe({
            "Depth (m)": res["z"],
            "σ'_v (kPa)": res["sigma_v_eff"],
            "u (kPa)": res["u"],
            "σ_h,a,total (kPa)": res["sigma_h_a_tot"],
            "σ_h,p,total (kPa)": res["sigma_h_p_tot"],
        })

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Based on Rankine Earth Pressure Theory")
