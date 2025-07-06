import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm, probplot

st.set_page_config(layout="wide")
st.title("Log-Normal Population Analyzer (PDF Mode)")

np.random.seed(0)

# === Sidebar ===
st.sidebar.header("Controls")
log_evaluation = st.sidebar.checkbox("View in logarithmic scale", value=True)
show_pop_controls = st.sidebar.checkbox("Show population parameter controls", value=False)

# === Population Parameters ===
if show_pop_controls:
    st.sidebar.header("Population Parameters")
    n_samples = st.sidebar.slider("Sample Size per Population", 300, 3000, 1000, step=100)

    mean1 = st.sidebar.slider("Log Mean of Pop 1", -1.0, 3.0, 0.5)
    std1 = st.sidebar.slider("Log Std Dev of Pop 1", 0.1, 1.5, 0.4)

    mean2 = st.sidebar.slider("Log Mean of Pop 2", -1.0, 3.0, 1.2)
    std2 = st.sidebar.slider("Log Std Dev of Pop 2", 0.1, 1.5, 0.3)

    mean3 = st.sidebar.slider("Log Mean of Pop 3", -1.0, 3.0, 2.0)
    std3 = st.sidebar.slider("Log Std Dev of Pop 3", 0.1, 1.5, 0.35)
else:
    n_samples = 1000
    mean1, std1 = 0.5, 0.4
    mean2, std2 = 1.2, 0.3
    mean3, std3 = 2.0, 0.35

# === Generate Populations ===
pop1 = np.random.lognormal(mean1, std1, n_samples)
pop2 = np.random.lognormal(mean2, std2, n_samples)
pop3 = np.random.lognormal(mean3, std3, n_samples)
combined = np.concatenate([pop1, pop2, pop3])
log_combined = np.log(combined)

# === Borders
combined_min, combined_max = float(np.min(combined)), float(np.max(combined))
st.sidebar.header("Adjust Borders")
border1 = st.sidebar.slider("Border 1", combined_min, combined_max, float(np.percentile(combined, 30)))
border2 = st.sidebar.slider("Border 2", combined_min, combined_max, float(np.percentile(combined, 70)))
if border1 >= border2:
    st.sidebar.error("Border 1 must be less than Border 2")

# === Masks
mask1 = combined < border1
mask2 = (combined >= border1) & (combined < border2)
mask3 = combined >= border2

g1, g2, g3 = combined[mask1], combined[mask2], combined[mask3]
g1_log, g2_log, g3_log = np.log(g1), np.log(g2), np.log(g3)

# === Stats Display
st.subheader("Group Statistics (log and original scales)")

def show_stats(data, label):
    if len(data) == 0:
        st.markdown(f"**{label}**: _No data_")
        return
    logd = np.log(data)
    st.markdown(f"""**{label}**
- Count: {len(data)}
- Mean (log): {np.mean(logd):.4f} | Std (log): {np.std(logd):.4f}
- Mean (original): {np.mean(data):.4f} | Std (original): {np.std(data):.4f}""")

col1, col2, col3 = st.columns(3)
with col1:
    show_stats(g1, "Group 1")
with col2:
    show_stats(g2, "Group 2")
with col3:
    show_stats(g3, "Group 3")

# === PDF Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
x_vals = np.logspace(np.log10(max(1e-3, np.min(combined))), np.log10(np.max(combined)), 1000)
x_log = np.linspace(np.min(log_combined), np.max(log_combined), 1000)

colors = ['blue', 'orange', 'green']
groups = [(g1, g1_log), (g2, g2_log), (g3, g3_log)]

# === PDF Plot
if log_evaluation:
    axs[0].set_xscale("log")
    for (g, _), color, label in zip(groups, colors, ['Group 1', 'Group 2', 'Group 3']):
        if len(g) > 2:
            s, loc, scale = lognorm.fit(g, floc=0)
            pdf = lognorm.pdf(x_vals, s, loc, scale)
            axs[0].plot(x_vals, pdf, label=label, color=color)

    s, loc, scale = lognorm.fit(combined, floc=0)
    axs[0].plot(x_vals, lognorm.pdf(x_vals, s, loc, scale), 'k--', label='Combined')
    axs[0].axvline(border1, color='black', linestyle='--')
    axs[0].axvline(border2, color='black', linestyle='--')
    axs[0].set_title("PDFs in Original Scale (Log-Normal)")
    axs[0].set_xlabel("Value")
else:
    for (_, g_log), color, label in zip(groups, colors, ['Group 1', 'Group 2', 'Group 3']):
        if len(g_log) > 2:
            mu, sigma = norm.fit(g_log)
            axs[0].plot(x_log, norm.pdf(x_log, mu, sigma), label=label, color=color)

    mu, sigma = norm.fit(log_combined)
    axs[0].plot(x_log, norm.pdf(x_log, mu, sigma), 'k--', label='Combined')
    axs[0].axvline(np.log(border1), color='black', linestyle='--')
    axs[0].axvline(np.log(border2), color='black', linestyle='--')
    axs[0].set_title("PDFs of Log-Transformed Data")
    axs[0].set_xlabel("Log(Value)")

axs[0].legend()

# === QQ Plot
def qq(data, label, color):
    if len(data) > 1:
        (osm, osr), (slope, intercept, _) = probplot(data, dist="norm")
        axs[1].plot(osm, osr, 'o', label=label, color=color, alpha=0.6)
        axs[1].plot(osm, slope * osm + intercept, linestyle='--', color=color, alpha=0.8)

qq(np.log(g1), "Group 1", "blue")
qq(np.log(g2), "Group 2", "orange")
qq(np.log(g3), "Group 3", "green")
qq(log_combined, "Combined", "black")

# Borders in QQ
sorted_comb = np.sort(log_combined)
n = len(sorted_comb)
p1 = np.searchsorted(sorted_comb, np.log(border1)) / n
p2 = np.searchsorted(sorted_comb, np.log(border2)) / n
q1 = norm.ppf(p1)
q2 = norm.ppf(p2)
axs[1].axvline(q1, color='black', linestyle='--')
axs[1].axvline(q2, color='black', linestyle='--')
axs[1].set_title("QQ Plot (Log-Transformed Data)")
axs[1].legend()

# === Show Plot
st.pyplot(fig)
