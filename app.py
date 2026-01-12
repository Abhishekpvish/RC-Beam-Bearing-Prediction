import streamlit as st
import math
import numpy as np
import tensorflow as tf
import joblib

# ====================================================
# LOAD WEIGHTED NEURAL NETWORK (INFERENCE ONLY)
# ====================================================
model = tf.keras.models.load_model("beam_weighted_model.keras")
scaler = joblib.load("scaler.save")

st.write("Scaler expects features:", scaler.n_features_in_)

def nn_predict(features):
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0][0]

# ====================================================
# IS 456 τc TABLE (p_t vs fck) — UNCHANGED
# ====================================================
tc_table = {
    20: [0.28, 0.32, 0.36, 0.40, 0.45],
    25: [0.29, 0.33, 0.37, 0.41, 0.46],
    30: [0.30, 0.34, 0.38, 0.42, 0.47],
    35: [0.31, 0.35, 0.39, 0.43, 0.48],
    40: [0.32, 0.36, 0.40, 0.44, 0.49]
}

pt_range = [0.15, 0.25, 0.50, 0.75, 1.0]
tc_max = {20: 2.8, 25: 3.1, 30: 3.5, 35: 3.7, 40: 4.0}

def get_tc(pt, fck):
    pt = max(min(pt, 1.0), 0.15)
    return float(np.interp(pt, pt_range, tc_table[fck]))

# ====================================================
# SELF WEIGHT — UNCHANGED
# ====================================================
def self_weight_kN_per_m(b, D):
    density = 25  # kN/m3
    return density * (b / 1000) * (D / 1000)

# ====================================================
# MAIN IS-456 CALCULATION FUNCTION — UNCHANGED
# ====================================================
def calculate(fck, fy, b, D, L, load_type,
              main_dia, main_count, stirrup_dia, spacing):

    cover = 25
    d = D - cover - stirrup_dia - main_dia / 2

    if d <= 0:
        return None, "Invalid effective depth"

    Ast = (math.pi / 4) * (main_dia ** 2) * main_count
    Asv = (math.pi / 4) * (stirrup_dia ** 2) * 2

    xu = (0.87 * fy * Ast) / (0.36 * fck * b)
    xu_max = 0.48 * d
    xu = min(xu, xu_max)

    Mu = 0.36 * fck * b * xu * (d - 0.42 * xu)
    Mu_lim = 0.138 * fck * b * d * d
    Mu = min(Mu, Mu_lim)

    eff_span = min(L + d, L)

    if load_type == "Point Load":
        W_flex = 4 * Mu / eff_span
    else:
        W_flex = 6 * Mu / eff_span

    pt = 100 * Ast / (b * d)
    tc = get_tc(pt, fck)
    tc_lim = tc_max[fck]

    V = W_flex / 2
    tau_v = V / (b * d)

    Vc = tc * b * d
    Vs = 0.87 * fy * Asv * d / spacing
    Vu = Vc + Vs

    W_shear = 2 * Vu
    Wu = min(W_flex, W_shear)

    sw = self_weight_kN_per_m(b, D) * (L / 1000)
    Wu_net = Wu / 1000 - sw

    if W_flex < 0.9 * W_shear:
        mode = "Flexural"
    elif W_shear < 0.9 * W_flex:
        mode = "Shear"
    else:
        mode = "Combined"

    warnings = []
    if tau_v > tc_lim:
        warnings.append("τv exceeds τc,max → unsafe section.")
    if Wu_net <= 0:
        warnings.append("Beam fails under self weight!")

    return {
        "Wu_kN_gross": Wu / 1000,
        "Wu_kN_net": Wu_net,
        "Mu_kNm": Mu / 1e6,
        "Vu_kN": Vu / 1000,
        "d_mm": d,
        "pt_percent": pt,
        "tau_v": tau_v,
        "tau_c": tc,
        "tau_c_max": tc_lim,
        "mode": mode,
        "warnings": warnings
    }, None

# ====================================================
# STREAMLIT UI
# ====================================================
st.title("🔧 RC Beam Bearing Capacity (IS-456 + Weighted NN)")

fck = st.selectbox("Concrete Grade (fck)", [20, 25, 30, 35, 40])
fy = st.selectbox("Steel Grade (fy)", [415, 500])

b = st.number_input("Beam Width b (mm)", 150, 1000, 230)
D = st.number_input("Overall Depth D (mm)", 200, 1000, 450)
L = st.number_input("Beam Length L (mm)", 500, 10000, 4000)

load_type = st.selectbox("Load Type", ["Point Load", "Two Point Load"])

main_dia = st.number_input("Main Bar Diameter (mm)", 8, 32, 16)
main_count = st.number_input("Number of Main Bars", 1, 8, 2)

stirrup_dia = st.number_input("Stirrup Diameter (mm)", 6, 12, 8)
spacing = st.number_input("Stirrup Spacing (mm)", 80, 300, 150)

# ====================================================
# BUTTON 1 — IS-456 RESULT
# ====================================================
if st.button("Calculate using IS-456"):
    result, err = calculate(
        fck, fy, b, D, L, load_type,
        main_dia, main_count, stirrup_dia, spacing
    )

    if err:
        st.error(err)
    else:
        st.success(f"GROSS Capacity: {result['Wu_kN_gross']:.2f} kN")
        st.success(f"NET Capacity: {result['Wu_kN_net']:.2f} kN")
        st.info(f"Failure Mode: {result['mode']}")

        st.write("### Detailed Results")
        st.write(f"Flexural Moment: {result['Mu_kNm']:.2f} kN·m")
        st.write(f"Shear Capacity Vu: {result['Vu_kN']:.2f} kN")
        st.write(f"Effective Depth d: {result['d_mm']:.1f} mm")
        st.write(f"Steel Ratio pₜ: {result['pt_percent']:.2f}%")
        st.write(f"τv: {result['tau_v']:.3f} MPa")
        st.write(f"τc: {result['tau_c']:.3f} MPa")
        st.write(f"τc,max: {result['tau_c_max']:.2f} MPa")

        for w in result["warnings"]:
            st.warning(w)

# ====================================================
# BUTTON 2 — WEIGHTED NEURAL NETWORK RESULT
# ====================================================
if st.button("Predict using Weighted Neural Network"):
    features = [
        fck, fy, b, D, L,
        1 if load_type == "Point Load" else 2,
        main_dia, main_count,
        stirrup_dia, spacing
    ]

    nn_value = nn_predict(features)
    st.success(f"NN Predicted NET Capacity: {nn_value:.2f} kN")

