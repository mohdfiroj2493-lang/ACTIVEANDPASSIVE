# 🏗️ Lateral Earth Pressure Calculator

A **Streamlit** web application for calculating active and passive lateral earth pressures using classical geotechnical methods — **Rankine**, **Coulomb**, and **At-Rest** — with full **surcharge pressure** support.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Rankine Theory** | Active & passive Ka/Kp, sloped backfill (β), cohesion (c) |
| **Coulomb Theory** | Wall friction (δ), wall batter (α), sloped backfill (β) |
| **At-Rest Pressure** | Jaky's formula with backfill slope correction |
| **Surcharge Types** | Uniform load, Line load, Point load (Boussinesq) |
| **Pressure Diagrams** | Side-by-side pressure diagrams for all 3 methods |
| **Comparison Charts** | Overlay charts, resultant force bar charts |
| **Formulas & Theory** | In-app reference with all equations |
| **Data Export** | Download results as CSV |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/lateral-earth-pressure.git
cd lateral-earth-pressure
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📐 Input Parameters

### Soil Properties
| Parameter | Symbol | Range | Default |
|---|---|---|---|
| Unit weight | γ (kN/m³) | 10–25 | 18 |
| Friction angle | φ (°) | 0–45 | 30 |
| Cohesion | c (kPa) | 0–100 | 0 |
| Wall friction | δ (°) | 0–35 | 15 |

### Wall Geometry
| Parameter | Symbol | Range | Default |
|---|---|---|---|
| Wall height | H (m) | 1–30 | 6 |
| Wall inclination | α (°) | -20 to +20 | 0 |
| Backfill slope | β (°) | 0–30 | 0 |

### Surcharge Options
- **None** – No surcharge
- **Uniform** – Constant distributed load q (kPa)
- **Line Load** – Linear load Q (kN/m) at distance x from wall
- **Point Load** – Concentrated load Q (kN) at distance x, depth z

---

## 📊 Theory & Formulas

### Rankine Active Coefficient
$$K_a = \tan^2\left(45° - \frac{\phi}{2}\right) = \frac{1 - \sin\phi}{1 + \sin\phi}$$

### Coulomb Active Coefficient
$$K_a = \frac{\cos^2(\phi - \alpha)}{\cos^2\alpha\cos(\delta+\alpha)\left[1+\sqrt{\frac{\sin(\phi+\delta)\sin(\phi-\beta)}{\cos(\delta+\alpha)\cos(\alpha-\beta)}}\right]^2}$$

### At-Rest (Jaky's Formula)
$$K_0 = 1 - \sin\phi$$

### Lateral Pressures
| Condition | Formula |
|---|---|
| Active (cohesionless) | σₐ = Kₐ γ z |
| Active (cohesive) | σₐ = Kₐ γ z − 2c√Kₐ |
| Passive | σₚ = Kₚ γ z + 2c√Kₚ |
| At-Rest | σ₀ = K₀ γ z |
| Uniform surcharge | Δσ = K q |

---

## 📁 Project Structure

```
lateral-earth-pressure/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🚀

---

## 📚 References

- Das, B.M. (2021). *Principles of Geotechnical Engineering*, 10th ed.
- Coduto, D.P. et al. (2011). *Geotechnical Engineering: Principles and Practices*
- Terzaghi, K., Peck, R.B. & Mesri, G. (1996). *Soil Mechanics in Engineering Practice*
- AASHTO LRFD Bridge Design Specifications

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
