"""Calculate the extinction strain rate of non-premixed counterflow diffusion flames at various pressures using Cantera."""
import sys
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, TypedDict
import numpy as np
import matplotlib.pyplot as plt

import cantera as ct


# TYPE ALIASES
# ------------
ThermoState = ct.composite.Solution
CounterflowFlame = ct.onedim.CounterflowDiffusionFlame

# CONSTANTS
# ---------
ATM_TO_PA = 101325.0
BAR_TO_PA = 1e05
R_U: Optional[float] = getattr(ct, "gas_constant", None)
if R_U is None:
    print("Error retrieving the universal gas constant from Cantera!")
    sys.exit(-1)
R_U = R_U * 1e-03
L = 5.45e-03
OUTPUT_PATH = Path() / "data"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
SOLUTION_FILE = OUTPUT_PATH / "methane_flame_data.h5"
SOLUTION_FILE.unlink(missing_ok=True)


class BoundaryConditions(TypedDict):
    """Dictionary for collecting boundary conditions."""

    a_g_init: float
    L: float
    P: float
    fuel: str
    T_f: float
    X_f: str
    T_ox: float
    X_ox: str


def names(entry: str) -> tuple[Path, str]:
    """Return the solution file and the desired entry name.

    :param entry: The desired entry name.

    :return: The solution file name and the desired entry name.
    :rtype: tuple[Path, str]
    """
    return (SOLUTION_FILE, entry)


def initialize(ii: int, rxn: str, bc: BoundaryConditions, grid: list[float]) -> tuple[ThermoState, CounterflowFlame, float]:
    """Initialize the counterflow extinction simulation.

    :param ii: Pressure iteration
    :param rxn: Reaction mechanism file
    :param bc: Boundary conditions of the problem
    :param grid: Meshing parameters

    :return: Cantera gas and counterflow simulation objects and the extinction limit temperature.
    :rtype: tuple[ThermoState, CounterflowFlame, float]
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

    f.set_grid_min(1e-7)
    f.set_max_grid_points(f.domains[f.domain_index("flame")], 1e4)
    f.set_refine_criteria(ratio=grid[0], slope=grid[1], curve=grid[2], prune=grid[3])

    temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T)

    print('Creating the initial solution ', end="", flush=True)
    f.solve(loglevel=0, auto=True)
    print('--> Done!')

    file_name, entry = names(f"p{ii+1}/initial-solution")
    f.save(file_name, name=entry, description="Initial solution", overwrite=True)
    f.save(f"./data/test{ii}.csv", overwrite=True)

    return (gas, f, temperature_limit_extinction)


def calculate_extinction(ii: int, p_fac: float, f: CounterflowFlame, T_lim: float) -> Tuple[float, Dict[str, float], int]:
    """Calculate the extinction strain rate iteratively.

    :param ii: Pressure iteration
    :param p_fac: Pressure factor
    :param f: Counterflow flame object.
    :param T_lim: Threshold extinction temperature.

    :return: Maximum temperature and strain-rates at extinction and last burning solution index
    :rtype: tuple[np.ndarray, np.ndarray, int]
    """
    # Exponents for the initial solution variation with changes in strain rate taken from Fiala and Sattelmayer (2014)
    exp_d_a = - 1. / 2.
    exp_d_p = -1. / 2.

    exp_u_a = 1. / 2.
    exp_u_p = -1. / 2.

    exp_V_a = 1.
    exp_V_p = 0.0

    exp_lam_a = 2.
    exp_lam_p = 1.

    exp_mdot_a = 1. / 2.
    exp_mdot_p = 1. / 2.

    # Set normalized initial strain rate
    alpha = [1.]
    # Initial relative strain rate increase
    delta_alpha = 1.
    # Factor of refinement of the strain rate increase
    delta_alpha_factor = 50.
    # Limit of the refinement: Minimum normalized strain rate increase
    delta_alpha_min = .001
    # Limit of the Temperature decrease
    delta_T_min = 1 # K

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
        f.flame.grid *= strain_factor ** exp_d_a * p_fac ** exp_d_p
        normalized_grid = f.grid / (f.grid[-1] - f.grid[0])  # the difference is essentially the width
        # Update mass fluxes
        f.fuel_inlet.mdot *= strain_factor ** exp_mdot_a * p_fac ** exp_mdot_p
        f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a * p_fac ** exp_mdot_p
        # Update velocities
        f.set_profile('velocity', normalized_grid,
                      f.velocity * strain_factor ** exp_u_a * p_fac ** exp_u_p)
        f.set_profile('spread_rate', normalized_grid,
                      f.spread_rate * strain_factor ** exp_V_a * p_fac ** exp_V_p)
        # Update pressure curvature
        f.set_profile('lambda', normalized_grid, f.L * strain_factor ** exp_lam_a * p_fac ** exp_lam_p)
        try:
            f.solve(loglevel=0, auto=False)
        except ct.CanteraError as e:
            print('Error: Did not converge at n =', n, e)

        T_max.append(np.max(f.T))
        a_max.append(f.strain_rate('max'))

        # CHECK FOR EXTINCTION
        if not np.isclose(np.max(f.T), T_lim):
            # Flame is still burning, so proceed to next strain rate
            n_last_burning = n
            file_name, entry = names(f"p{ii+1}/extinction/{n:04d}")
            f.save(file_name, name=entry, description=f"Solution at alpha = {alpha[-1]}",
                   overwrite=True)

            print('Flame burning at alpha = {:8.4F}. Proceeding to the next iteration, '
                  'with delta_alpha = {}'.format(alpha[-1], delta_alpha))
        elif ((T_max[-2] - T_max[-1] < delta_T_min) and (delta_alpha < delta_alpha_min)):
            # If the temperature difference is too small and the minimum relative
            # strain rate increase is reached, save the last, non-burning, solution
            # to the output file and break the loop
            file_name, entry = names(f"p{ii+1}/extinction/{n:04d}")
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
            file_name, entry = names(f"p{ii+1}/extinction/{n_last_burning:04d}")
            f.restore(file_name, entry)

    print(f"EXTINCTION SIMULATION COMPLETED FOR PRESSURE {ii+1}!")
    print("-----------------------------------------------------")

    file_name, entry = names(f"p{ii+1}/extinction/{n_last_burning:04d}")
    f.restore(file_name, entry)

    T_ext = np.max(f.T)
    a_ext = {
        "mean": f.strain_rate('mean'),
        "max": f.strain_rate('max'),
        "pot_f": f.strain_rate('potential_flow_fuel'),
        "pot_ox": f.strain_rate('potential_flow_oxidizer'),
        "stoich": f.strain_rate('stoichiometric', fuel=bc["fuel"])
    }

    return (T_ext, a_ext, n_last_burning)


def plot_data():
    pass
    # fig, ax1 = plt.subplots()

    # # Plot the first dataset on the first y-axis (ax1)
    # ax1.semilogx(p, T_ext, 'b-')
    # ax1.set_xlabel('p')
    # ax1.set_ylabel('T_max', color='b')
    # ax1.tick_params(axis='y', labelcolor='b')

    # # Create a second y-axis that shares the same x-axis
    # ax2 = ax1.twinx()


    # a_plot = [x["mean"] for x in a_ext]

    # # Plot the second dataset on the second y-axis (ax2)
    # ax2.semilogx(p, a_plot, 'r-')
    # ax2.set_ylabel('a_max', color='r')
    # ax2.tick_params(axis='y', labelcolor='r')

    # # Set the second y-axis to logarithmic scale
    # ax2.set_yscale('log')

    # # Show the plot
    # plt.show()


if __name__ == "__main__":
    # Modes
    calcExt = True
    plotDat = False

    # Inputs
    rxn = 'FFCM-2/FFCM-2.yaml'
    pressures = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]) * BAR_TO_PA
    bc: BoundaryConditions = {
        "a_g_init": 80,
        "L": L,
        "P": pressures[0],
        "fuel": "CH4",
        "T_f": 300.0,
        "X_f": "CH4:1.0",
        "T_ox": 300.0,
        "X_ox": "O2:0.21 N2:0.78 AR:0.01"
    }
    grid = [3.0, 0.2, 0.2, 0.03]

    # Caclulation
    if calcExt:
        TT = []
        AA = []
        NN = []

        for ii, p in enumerate(pressures):
            bc["P"] = p

            if ii == 0:
                p_fac = 1.0
                a_fac = 1.0
            else:
                p_fac = p / pressures[ii - 1]
                a_g_init_old = bc["a_g_init"]
                bc["a_g_init"] = bc["a_g_init"] * p_fac**(3.0 / 2.0)
                a_fac = bc["a_g_init"] / a_g_init_old
                bc["L"] = bc["L"] * a_fac ** (-1.0 / 2.0) * p_fac ** (-1.0 / 2.0)

            # Simulation
            gas, f, T_lim = initialize(ii, rxn, bc, grid)
            T_ext, a_ext, n_last_burning = calculate_extinction(ii, p_fac, f, T_lim)

            TT.append(T_ext)
            AA.append(a_ext)
            NN.append(n_last_burning)

        with open("./data/extinction.pkl", "wb") as fle:
            pickle.dump((pressures, TT, AA, NN), fle)

        print(TT)

    # Plotting
    if plotDat:
        plot_data()
