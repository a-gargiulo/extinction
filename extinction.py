"""Calculate the extinction strain rate of non-premixed counterflow diffusion flames at various pressures using Cantera."""

from typing import TypedDict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import cantera as ct


class BoundaryConditions(TypedDict):
    a_g: float
    L: float
    P: float
    f: str
    f_d: str
    T_f: float
    X_f: float
    ox: str
    ox_d: str
    T_ox: float
    X_ox: float


# Constants
ATM_TO_PA = 101325.0
SEC_TO_MIN = 1.0 / 60.0
M3_TO_L = 1000.0
R_U = 8.314

# Counterflow burner geometry
D_i = 0.0065
A_i = D_i**2 * np.pi / 4
A_o = np.pi * ((0.01495 / 2) ** 2 - (0.00726 / 2) ** 2)
L = 5.45e-3
a_g = 100

M = {
    "air": 28.97e-03,
    "ethylene": 28.05e-03,
    "hydrogen": 1.00e-03,
    "methane": 16.04e-03,
    "nitrogen": 28.00e-03,
    "oxygen": 32.00e-03,
    "helium": 4.00e-03,
}

# Initialization
# --------------
reaction_mechanism = 'FFCM-2/FFCM-2.yaml'
gas = ct.Solution(reaction_mechanism)
f = ct.CounterflowDiffusionFlame(gas, width=L)

f.P = 1.0e5

fuel = "methane"
fuel_diluent = "nitrogen"
X_f = 0.15
f.fuel_inlet.X = 'CH4:1'
f.fuel_inlet.T = 300

ox = "oxygen"
ox_diluent = "nitrogen"
X_ox = 0.21
f.oxidizer_inlet.X = 'O2:0.21 N2:0.79'
f.oxidizer_inlet.T = 300


v_ox_total = L * a_g / 4
M_ox_total = X_ox * M[ox.lower()] + (1 - X_ox) * M[ox_diluent.lower()]
rho_ox_total = f.P * M_ox_total / R_U / f.oxidizer_inlet.T

M_f_total = X_f * M[fuel.lower()] + (1 - X_f) * M[fuel_diluent.lower()]
rho_f_total = f.P * M_f_total / R_U / f.fuel_inlet.T
v_f_total = v_ox_total * float(np.sqrt(rho_ox_total / rho_f_total))

# Define the operating pressure and boundary conditions
# f.fuel_inlet.mdot = 0.5  # kg/m^2/s
f.fuel_inlet.mdot = v_f_total * rho_f_total  # kg/m^2/s
f.oxidizer_inlet.mdot = v_ox_total * rho_ox_total  # kg/m^2/s

# Set refinement parameters
ratio = 3.0
slope = 0.1
curve = 0.2
prune = 0.03
f.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)

# Define a limit for the maximum temperature below which the flame is
# considered as extinguished and the computation is aborted
temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T)

# Initialize and solve
print('Creating the initial solution')
f.solve(loglevel=0, auto=True)

# Define output locations
output_path = Path() / "diffusion_flame_extinction_data"
output_path.mkdir(parents=True, exist_ok=True)

hdf_output = "native" in ct.hdf_support()
if hdf_output:
    file_name = output_path / "flame_data.h5"
    file_name.unlink(missing_ok=True)

def names(test):
    if hdf_output:
        # use internal container structure for HDF
        file_name = output_path / "flame_data.h5"
        return file_name, test
    # use separate files for YAML
    file_name = output_path / f"{test}.yaml".replace("-", "_").replace("/", "_")
    return file_name, "solution"

file_name, entry = names("initial-solution")
f.save(file_name, name=entry, description="Initial solution", overwrite=True)

# %%
# Compute Extinction Strain Rate
# ------------------------------
# Exponents for the initial solution variation with changes in strain rate
# Taken from Fiala and Sattelmayer (2014)
exp_d_a = - 1. / 2.
exp_u_a = 1. / 2.
exp_V_a = 1.
exp_lam_a = 2.
exp_mdot_a = 1. / 2.

# Set normalized initial strain rate
alpha = [1.]
# Initial relative strain rate increase
delta_alpha = 1.
# Factor of refinement of the strain rate increase
delta_alpha_factor = 50.
# Limit of the refinement: Minimum normalized strain rate increase
delta_alpha_min = .001
# Limit of the Temperature decrease
delta_T_min = 1  # K

# Iteration indicator
n = 0
# Indicator of the latest flame still burning
n_last_burning = 0
# List of peak temperatures
T_max = [np.max(f.T)]
# List of maximum axial velocity gradients
a_max = [np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))]

# %%
# Simulate counterflow flames at increasing strain rates until the flame is
# extinguished. To achieve a fast simulation, an initial coarse strain rate
# increase is set. This increase is reduced after an extinction event and
# the simulation is again started based on the last burning solution.
# The extinction point is considered to be reached if the abortion criteria
# on strain rate increase and peak temperature decrease are fulfilled.
while True:
    n += 1
    # Update relative strain rates
    alpha.append(alpha[n_last_burning] + delta_alpha)
    strain_factor = alpha[-1] / alpha[n_last_burning]
    # Create an initial guess based on the previous solution
    # Update grid
    # Note that grid scaling changes the diffusion flame width
    f.flame.grid *= strain_factor ** exp_d_a
    normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
    # Update mass fluxes
    f.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
    f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
    # Update velocities
    f.set_profile('velocity', normalized_grid,
                  f.velocity * strain_factor ** exp_u_a)
    f.set_profile('spread_rate', normalized_grid,
                  f.spread_rate * strain_factor ** exp_V_a)
    # Update pressure curvature
    f.set_profile('lambda', normalized_grid, f.L * strain_factor ** exp_lam_a)
    try:
        f.solve(loglevel=0)
    except ct.CanteraError as e:
        print('Error: Did not converge at n =', n, e)

    T_max.append(np.max(f.T))
    a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))
    if not np.isclose(np.max(f.T), temperature_limit_extinction):
        # Flame is still burning, so proceed to next strain rate
        n_last_burning = n
        file_name, entry = names(f"extinction/{n:04d}")
        f.save(file_name, name=entry, description=f"Solution at alpha = {alpha[-1]}",
               overwrite=True)

        print('Flame burning at alpha = {:8.4F}. Proceeding to the next iteration, '
              'with delta_alpha = {}'.format(alpha[-1], delta_alpha))
    elif ((T_max[-2] - T_max[-1] < delta_T_min) and (delta_alpha < delta_alpha_min)):
        # If the temperature difference is too small and the minimum relative
        # strain rate increase is reached, save the last, non-burning, solution
        # to the output file and break the loop
        file_name, entry = names(f"extinction/{n:04d}")
        f.save(file_name, name=entry, overwrite=True,
               description=f"Flame extinguished at alpha={alpha[-1]}")

        print('Flame extinguished at alpha = {0:8.4F}.'.format(alpha[-1]),
              'Abortion criterion satisfied.')
        break
    else:
        # Procedure if flame extinguished but abortion criterion is not satisfied
        # Reduce relative strain rate increase
        delta_alpha = delta_alpha / delta_alpha_factor

        print('Flame extinguished at alpha = {0:8.4F}. Restoring alpha = {1:8.4F} and '
              'trying delta_alpha = {2}'.format(
                  alpha[-1], alpha[n_last_burning], delta_alpha))

        # Restore last burning solution
        file_name, entry = names(f"extinction/{n_last_burning:04d}")
        f.restore(file_name, entry)

# %%
# Results
# -------
# Print some parameters at the extinction point, after restoring the last burning
# solution
file_name, entry = names(f"extinction/{n_last_burning:04d}")
f.restore(file_name, entry)

print('----------------------------------------------------------------------')
print('Parameters at the extinction point:')
print('Pressure p={0} bar'.format(f.P / 1e5))
print('Peak temperature T={0:4.0f} K'.format(np.max(f.T)))
print('Mean axial strain rate a_mean={0:.2e} 1/s'.format(f.strain_rate('mean')))
print('Maximum axial strain rate a_max={0:.2e} 1/s'.format(f.strain_rate('max')))
print('Fuel inlet potential flow axial strain rate a_fuel={0:.2e} 1/s'.format(
      f.strain_rate('potential_flow_fuel')))
print('Oxidizer inlet potential flow axial strain rate a_ox={0:.2e} 1/s'.format(
      f.strain_rate('potential_flow_oxidizer')))
print('Axial strain rate at stoichiometric surface a_stoich={0:.2e} 1/s'.format(
      f.strain_rate('stoichiometric', fuel='H2')))

# %%
# Plot the maximum temperature over the maximum axial velocity gradient
plt.figure()
plt.semilogx(a_max, T_max)
plt.xlabel(r'$a_{max}$ [1/s]')
plt.ylabel(r'$T_{max}$ [K]')
plt.savefig(output_path / "figure_T_max_a_max.png")
plt.show()


def init_mass_flux(bc: BoundaryConditions):
    """Calculate the initial oxidizer and fuel mass flux for a set of boundary conditions."""
    v_ox_total = bc["L"] * bc["a_g"] / 4
    M_ox_total = bc["X_ox"] * M[bc["ox"].lower()] + (1 - bc["X_ox"]) * M[bc["ox_d"].lower()]
    rho_ox_total = bc["P"] * M_ox_total / R_U / bc["T_ox"]

    M_f_total = X_f * M[fuel.lower()] + (1 - X_f) * M[fuel_diluent.lower()]
    rho_f_total = f.P * M_f_total / R_U / f.fuel_inlet.T
    v_f_total = v_ox_total * float(np.sqrt(rho_ox_total / rho_f_total))

    # Define the operating pressure and boundary conditions
    # f.fuel_inlet.mdot = 0.5  # kg/m^2/s
    f.fuel_inlet.mdot = v_f_total * rho_f_total  # kg/m^2/s
    f.oxidizer_inlet.mdot = v_ox_total * rho_ox_total  # kg/m^2/s



