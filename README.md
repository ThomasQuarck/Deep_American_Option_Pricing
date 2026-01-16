# Deep Learning for American Option Pricing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Description
This repository implements a **Deep Learning pricer** for an American Option (Bermudan style).

The goal was to replicate the **Longstaff-Schwartz algorithm** but replace the traditional polynomial regression basis (Laguerre/Legendre) with a **Deep Neural Network**. This approach is particularly useful for high-dimensional problems where standard regression methods struggle.

The code is implemented in **Python** using **PyTorch** for GPU acceleration.

## Methodology
The pricing engine treats the early exercise problem as an Optimal Stopping problem solvable via Dynamic Programming (Backward Induction):

1.  **Simulation:** 100,000 asset paths generated via Geometric Brownian Motion.
2.  **Regression:** At each time step $t$, a Feed-Forward Neural Network approximates the **Continuation Value** (expected future cashflows).
3.  **Optimization:** The model compares the immediate payoff $h(S_t) = (K - S_t)^+$ with the network's prediction to determine the optimal exercise boundary.

$$V_t(S_t) = \max \left( h(S_t), \mathbb{E}[ e^{-rdt} V_{t+1} | S_t ] \right)$$

## Technical Implementation
* **Framework:** PyTorch (chosen for its dynamic computation graph and easy GPU casting).
* **Architecture:** Simple MLP (3 hidden layers, ReLU activation) sufficient to capture the convex payoff structure.
* **Performance:** The code is fully vectorized. Pricing 100k paths takes a few seconds on a T4 GPU.

## Results
Tested on a standard Put Option ($S_0=100$, $K=110$, $r=10\%$, $\sigma=25\%$).

* **Converged Price:** ~11.97
* **Consistency:** The result aligns with standard literature values for American Puts with these parameters.

## Potential Applications
While developed for financial derivatives, this logic applies to other **Optimal Stopping** problems:
* **Parametric Insurance:** Modeling weather triggers.
* **Energy Storage:** Optimizing battery charge/discharge cycles based on spot prices.

## Future Improvements
* Extending the model to **Multi-Asset Basket Options** to demonstrate the high-dimensional advantage.
* Implementing a **Deep Hedging** module to minimize PnL variance.
* Adding a direct comparison benchmark (e.g., Binomial Tree or standard LSM).

---
**Author:** [Thomas] [Quarck]
