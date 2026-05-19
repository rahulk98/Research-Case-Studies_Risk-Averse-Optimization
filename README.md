# Research Case Studies: Risk-Averse Optimization

Three case studies exploring Conditional Value-at-Risk (CVaR) as a drop-in replacement for expected-value objectives, ranging from a toy nonconvex landscape to a real bioreactor control problem. Full write-up lives in `Scientific Report/scientificthesis.pdf`.

## Repository structure

```
Case Study 1 - Wobbly Rosenbrock/                       Toy nonconvex problem with parameter noise
Case Study 2 - SVM on Perturbed Cats vs Dogs/           Binary image classifier under input perturbations
Case Study 3 - Risk Averse Optimal Control for Wine     Nonlinear MPC for a wine-fermentation bioreactor
Fermentation/
Scientific Report/                                       LaTeX source and compiled PDF of the full thesis
```

## Case studies

### 1. Wobbly Rosenbrock
Minimizes CVaR of the Rosenbrock loss when the shape parameter is perturbed by Gaussian noise. Used as a sanity check: the nominal minimum already sits in a locally robust basin, so CVaR barely moves the optimizer. Notebook: `wobbly_rosenbrock_cvar_param_noise.ipynb`.

### 2. Robust SVM on perturbed cats vs dogs
Binary SVM on 64x64 grayscale images, trained by minimizing CVaR of the hinge loss across perturbed copies of the dataset. Modest accuracy gain (54.33% to 56.33%) and a visibly less jittery decision boundary under pixel noise. Notebook: `cvar_param_svm.ipynb`.

### 3. Wine fermentation MPC
Six-state ODE bioreactor model (biomass, nitrogen, ethanol, sugar, oxygen, reactor temperature) controlled over a 21-day horizon. Nominal MPC plans against a single cellar-temperature forecast; CVaR MPC optimizes against 50 random cellar trajectories at alpha = 0.05. Stress-tested on 100 volatile cellar profiles (14 to 28 degrees C).

| Metric                | Nominal MPC | CVaR MPC |
| --------------------- | ----------- | -------- |
| Constraint violations | 464         | 0        |
| Failed scenarios      | 81%         | 0%       |
| Final sugar (g/L)     | 10.31       | 7.60     |
| Control effort        | 7.69        | 38.52    |

Notebooks: `MPC_Freddo_CVaR_Cellar_Temp - Direct Comparison.ipynb`, `MPC_Freddo_CVaR_Cellar_Temp - Simulation Comparison.ipynb`. Plots in `part 3 result plots/`.

## Stack

Python throughout. CasADi and IPOPT for the nonlinear MPC, CVXPY and OSQP for the SVM QPs, NumPy and Matplotlib for plotting. Each case-study directory contains a notebook that reproduces its plots.

## Reference

Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2, 21-41.
