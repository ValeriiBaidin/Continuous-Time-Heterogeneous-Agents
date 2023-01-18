# Continuous Time Heterogeneous Agent models:

The repository consists of a list of heterogeneous agent models.

The following models are based on Benjamin Moll's Matlab code (https://benjaminmoll.com/codes/)

1. Hugget_model.jl - HJB Equation implicit method, Kolmogorov Forward Equation, and Equilibrium interest rate with the non-linear solver.

2. HJB_Labor_supply.jl - Consumption-Saving Problem with Endogenous Labor Supply

3. HANK_lite.jl - HANK is a lite replication of HANK (Kaplan et al. 2018). The file consists only of the solution for equilibrium and doesn't transition dynamics. **Learning rate was added**, thus algorithm converges almost with any input parameters. 
