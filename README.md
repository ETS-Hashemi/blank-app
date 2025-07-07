# Log-Normal Population Analyzer

This project provides an interactive framework for analyzing and visualizing overlapping **log-normal populations**, such as those encountered in **geochemical data interpretation**, **environmental monitoring**, or **economic modeling**.

The tool is specifically designed to support:

- Visual separation of mixed populations using dynamic thresholds
- Probability density estimation and classification
- Statistical comparison of log-transformed and original data
- Probability plots for assessing normality in logarithmic space

---

## 🔬 Scientific Motivation

In geochemical exploration and environmental data analysis, element concentrations often follow **log-normal distributions** due to multiplicative geological processes. Properly separating overlapping geochemical populations is essential for:

- Identifying geochemical anomalies or halos
- Defining background vs. anomaly thresholds
- Supporting machine learning classification
- Interpreting probabilistic models

This tool assists in **manually and visually identifying those populations**, while computing descriptive statistics in both linear and log space.

---

## 📊 Features

- Simulates **3 log-normal populations** with tunable parameters (log-mean, log-std)
- Allows **manual separation** via border sliders
- Automatically updates:
  - Group-wise statistics (mean, std) in both original and log scales
  - Log-normal PDF curves
  - Combined population curve
  - Normal probability (QQ) plot of log-transformed data
- Switch between **original** and **logarithmic evaluation** modes

---

## 📁 Usage

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
