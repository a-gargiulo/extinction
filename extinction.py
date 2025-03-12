"""Calculate the extinction strain rate of non-premixed counterflow diffusion flames at various pressures using Cantera."""
import sys
from pathlib import Path
from typing import cast, TypedDict
import numpy as np
import matplotlib.pyplot as plt

import cantera as ct


# CONSTANTS
# ---------
ATM_TO_PA = 101325.0
R_U = ct.gas_constant * 1e-03
L = 5.45e-03

OUTPUT_PATH = Path() / "data"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

HDF_OUTPUT = "native" in ct.hdf_support()
if HDF_OUTPUT:
    file_name = OUTPUT_PATH / "flame_data.h5"
    file_name.unlink(missing_ok=True)

def names(test: str) -> tuple[Path, str]:
    if HDF_OUTPUT:
        # use internal container structure for HDF
        file_name = OUTPUT_PATH / "flame_data.h5"
        if file_name is None:
            sys.exit(-1)
        return (file_name, test)
    sys.exit(-1)


class BoundaryConditions(TypedDict):
    """Dictionary for collecting boundary conditions efficiently."""

    a_g_init: float
    L: float
    P: float
    T_f: float
    X_f: str
    T_ox: float
    X_ox: str


def initialize(
    rxn: str,
    bc: BoundaryConditions,
    grid: list[float]
) -> tuple[ct.composite.Solution, ct.onedim.CounterflowDiffusionFlame, float]:
    """Initialize the counterflow extinction simulation.

    :param rxn: Reaction mechanism file.
    :param bc: Boundary conditions of the problem.
    :param grid: Meshing parameters.

    :return: Cantera gas and counterflow simulation objects and the extinction limit temperature.
    :rtype: tuple[ct.composite.Solution, ct.onedim.CounterflowDiffusionFlame, float]
    """
    gas = ct.composite.Solution(rxn)
    f = ct.CounterflowDiffusionFlame(gas, width=L)

    f.P = bc["P"]

    f.fuel_inlet.T = bc["T_f"]
    f.fuel_inlet.X = bc["X_f"]

    f.oxidizer_inlet.T = bc["T_ox"]
    f.oxidizer_inlet.X = bc["X_ox"]

    v_ox_tot = bc["L"] * bc["a_g_init"] / 4.0
    M_ox_tot = np.sum(f.oxidizer_inlet.X @ gas.molecular_weights) * 1e-03
    rho_ox_tot = bc["P"] * M_ox_tot / R_U / bc["T_ox"]

    M_f_tot = np.sum(f.fuel_inlet.X @ gas.molecular_weights) * 1e-03
    rho_f_tot = bc["P"] * M_f_tot / R_U / bc["T_f"]
    v_f_tot = v_ox_tot * np.sqrt(rho_ox_tot / rho_f_tot)

    f.fuel_inlet.mdot = v_f_tot * rho_f_tot
    f.oxidizer_inlet.mdot = v_ox_tot * rho_ox_tot

    f.set_refine_criteria(ratio=grid[0], slope=grid[1], curve=grid[2], prune=grid[3])

    temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T)

    print('Creating the initial solution ', end="", flush=True)
    f.solve(loglevel=0, auto=True)
    print('--> Done!')

    file_name, entry = names("initial-solution")
    f.save(file_name, name=entry, description="Initial solution", overwrite=True)

    return (gas, f, temperature_limit_extinction)


def calculate_extinction_strain_rate(
    f: ct.onedim.CounterflowDiffusionFlame,
    T_lim: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Calculate the extinction strain rate iteratively.

    :param f: Counterflow flame object.
    :param T_lim: Threshold extinction temperature.

    :return: Maximum strain-rate and temperature and last burning solution index.
    :rtype: tuple[np.ndarray, np.ndarray, int]
    """
    # Exponents for the initial solution variation with changes in strain rate taken from Fiala and Sattelmayer (2014)
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
    # a_max = [np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))]
    a_max = [f.strain_rate('max')]

    while True:
        n += 1
        # Update relative strain rates
        alpha.append(alpha[n_last_burning] + delta_alpha)
        strain_factor = alpha[-1] / alpha[n_last_burning]  # this is essentially (a_new/a_old), since a[-1] was just appended
        # Create an initial guess based on the previous solution
        # Update grid
        # Note that grid scaling changes the diffusion flame width
        f.flame.grid *= strain_factor ** exp_d_a
        normalized_grid = f.grid / (f.grid[-1] - f.grid[0])  # the difference is essentially the width
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
        # a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))
        a_max.append(f.strain_rate('max'))
        # CHECK FOR EXTINCTION
        if not np.isclose(np.max(f.T), T_lim):
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
                  'trying delta_alpha = {2}'.format(alpha[-1], alpha[n_last_burning], delta_alpha))

            # Restore last burning solution
            file_name, entry = names(f"extinction/{n_last_burning:04d}")
            f.restore(file_name, entry)

    print("EXTINCTION SIMULATION COMPLETED!")
    print("--------------------------------")
    return (np.array(a_max), np.array(T_max), n_last_burning)


def print_results(f: ct.onedim.CounterflowDiffusionFlame, n_last_burning: int):
    """Print useful extinction information.

    :param f: Cantera counterflow flame object.
    :param n_last_burning: Last burning solution index.
    """
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
          f.strain_rate('stoichiometric', fuel='CH4')))


if __name__ == "__main__":
    # Inputs
    rxn = 'FFCM-2/FFCM-2.yaml'
    # rxn = 'gri30.yaml'
    p = [x * ATM_TO_PA for x in [1.0, 2.0, 4.0, 8.0, 10.0]]
    bc: BoundaryConditions = {
        "a_g_init": 200,
        "L": L,
        "P": 0.0,
        "T_f": 300.0,
        "X_f": "CH4:1",
        "T_ox": 300.0,
        "X_ox": "O2:0.21 N2:0.78 AR:0.01"
    }
    grid = [3.0, 0.2, 0.2, 0.03]

    for ii, pp in enumerate(p):
        bc["P"] = pp

        # Simulation
        gas, f, T_lim = initialize(rxn, bc, grid)
        a_max, T_max, n_last_burning = calculate_extinction_strain_rate(f, T_lim)
        print_results(f, n_last_burning)

        # Plot the maximum temperature over the maximum axial velocity gradient
        plt.figure()
        plt.semilogx(a_max, T_max, marker='o')
        plt.xlabel(r'$a_{max}$ [1/s]')
        plt.ylabel(r'$T_{max}$ [K]')
        plt.show()
