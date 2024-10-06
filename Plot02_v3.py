import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Plot01_V1 import calculate_forces, calculate_power, simulate_soc_and_power
from scipy.optimize import minimize_scalar
import os

# Function to load and initialize data
def load_data(file_path):
    data = pd.DataFrame({
        "M_tank": [5.6],  # kg
        "P_H2": [700],  # bar
        "Molar_mass_dihydrogen": [2.016],  # g/mol
        "LHV": [242],  # kJ/mol
        "HHV": [285],  # kJ/mol
        "H2_loss": [2],  # %
        "Faraday_const": [96487],  # C/mol
        "Air_stoich": [1.5],
        "P_atm": [1],  # bar
        "O2_molar_fraction_ambient_air": [21 / 100],  # %
        "Air_compressor_efficiency": [60 / 100],  # %
        "N_cell": [330],
        "Area_stack": [273],  # cm^2
        "Gamma": [1.4],
        "R": [8.314],  # J/mol.K (Ideal gas constant)
        "Ambient_temperature": [298]  # K
    })
    
    df = pd.read_excel(file_path, header=0)
    return data, df

# Function to calculate compressor power
def calculate_compressor_power(df, data):
    P_com = np.zeros(len(df))
    for i in range(len(df)):
        P_com[i] = (1 / data['Air_compressor_efficiency'][0]) * \
                    (data['Air_stoich'][0] / data['O2_molar_fraction_ambient_air'][0]) * \
                    (data['N_cell'][0] * df['i (A/cm²)'][i]) / (4 * data['Faraday_const'][0]) * \
                    (data['Gamma'][0] / (data['Gamma'][0] - 1)) * \
                    (data['R'][0] * data['Ambient_temperature'][0]) * \
                    ((df['P air (bar)'][i] / data['P_atm'][0])**((data['Gamma'][0] - 1) / data['Gamma'][0]) - 1) * data['Area_stack'][0] / 1000
    return P_com

# Function to calculate fuel cell power
def calculate_fuel_cell_power(df, data):
    P_fuel = np.zeros(len(df))
    for j in range(len(df)):
        P_fuel[j] = df['U_cell (V)'][j] * df['i (A/cm²)'][j] * data['N_cell'][0] * data['Area_stack'][0]
    return P_fuel

# Function to calculate hydrogen consumption
def calculate_hydrogen_consumption(df, data):
    Hydro_com = np.zeros(len(df))
    for i in range(len(df)):
        molar_flow = df['i (A/cm²)'][i] / (2 * data['Faraday_const'][0])
        Hydro_com[i] = molar_flow * data['Molar_mass_dihydrogen'][0] * data['N_cell'][0] * data['Area_stack'][0]
    return Hydro_com

# Function to create interpolation for hydrogen consumption
def fit_polynomial_hydrogen_consumption(df, Hydro_com):
    coefficients = np.polyfit(df['i (A/cm²)'], Hydro_com, 5)
    polynomial = np.poly1d(coefficients)
    i_fit = np.linspace(min(df['i (A/cm²)']), max(df['i (A/cm²)']), 1000)
    hydro_fit = polynomial(i_fit)
    return i_fit, hydro_fit

# Function to calculate WLTC hydrogen consumption
def calculate_wltc_hydrogen_consumption(WLTC_file, df):
    WLTC = pd.read_excel(WLTC_file, sheet_name='Sheet1', header=0)
    speed = WLTC.iloc[:, 1].values
    time = WLTC.iloc[:, 0].values

    current_density = df['i (A/cm²)'].values
    interp_function = interp1d(np.linspace(0, time[-1], len(current_density)), current_density, kind='linear', fill_value='extrapolate')
    interpolated_current_density = interp_function(time)

    return time, speed, interpolated_current_density

# Function to compute optimization results
def compute_results_optimization(time, Power_demand, A_cell, polarization_curve, N_cells, lb, ub, threshold):
    # Optimization routine as in the script...
    # Ensure lb and ub are scalars
    results = []
    interpolation_function = create_interpolation(polarization_curve)
    for t in range(len(time)):
        P_demand = Power_demand[t]
        iteration = 0
        U_cell = 0.6
        while True:
            def func(U_cell_guess):
                I_stack = calculate_current(P_demand, U_cell_guess)
                i_t = calculate_current_density(I_stack, A_cell)
                interpolated_value = interpolation_function(i_t.item())
                return (U_cell_guess - interpolated_value.item()) **2
            res = minimize_scalar(func, bounds=(lb, ub), method='bounded')

            if res.success:
                U_cell_new = res.x
                if abs(U_cell_new - U_cell) < threshold or iteration > 1000:
                    break
                U_cell = U_cell_new
                iteration += 1
            else:
                break

        I_stack = calculate_current(P_demand, U_cell)
        i_t = calculate_current_density(I_stack, A_cell)
        results.append({'Time': t, 'P_demand': P_demand, 'I_stack': I_stack, 'i_t': i_t, 'U_cell': U_cell})

    return pd.DataFrame(results)

# Function to calculate total hydrogen consumption and efficiency
def calculate_hydrogen_and_efficiency(time, Result_optimization, data):
    I_cell = np.zeros(len(time))
    mass_hydro = np.zeros(len(time))
    
    for i in range(len(time)):
        P_demand = Result_optimization['P_demand'][i]
        U_cell = Result_optimization['U_cell'][i]
        if P_demand < 0:
            mass_hydro[i] = 0
        else:
            if U_cell > 0:
                I_cell[i] = P_demand / (U_cell * data['N_cell'][0])
            else:
                I_cell[i] = 0
            mass_hydro[i] = ((I_cell[i] * data['N_cell'][0] * data['Molar_mass_dihydrogen'][0]) / (2 * data['Faraday_const'][0])) / 1000
    
    total_hydrogen_consumed = np.sum(mass_hydro)
    
    LHV_hydrogen = data['LHV'][0] * 1000
    Molar_mass_h2 = data['Molar_mass_dihydrogen'][0] / 1000
    efficiency = np.zeros(len(time))

    for i in range(len(time)):
        P_fuel_cell = Result_optimization['P_demand'][i]
        if mass_hydro[i] > 0:
            energy_from_h2 = mass_hydro[i] * LHV_hydrogen / Molar_mass_h2
            efficiency[i] = P_fuel_cell / energy_from_h2 if energy_from_h2 > 0 else 0
        else:
            efficiency[i] = efficiency[i-1]

    average_efficiency = np.mean(efficiency) * 100
    return total_hydrogen_consumed, average_efficiency, mass_hydro, efficiency

# Main function to execute all steps
def main():
    # Load data
    data, df = load_data('Plot02.xlsx')

    # Calculate compressor power
    P_com = calculate_compressor_power(df, data)

    # Plot compressor power
    plt.figure(figsize=(6, 6))
    plt.plot(df['U_cell (V)'], P_com, 'y-.', label='Power of Air Compressor')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Compressor Power (kW)')
    plt.legend()
    plt.savefig('Power_of_Air_compressor.png', dpi=200)
    plt.show()

    # Calculate fuel cell power
    P_fuel = calculate_fuel_cell_power(df, data)

    # Plot fuel cell power
    plt.figure(figsize=(6, 6))
    plt.plot(df['U_cell (V)'], P_fuel / 1000, 'b-.', label='Power by Fuel Cell')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.savefig('Power_of_Fuel_Cell.png', dpi=200)
    plt.show()

    # Hydrogen consumption
    Hydro_com = calculate_hydrogen_consumption(df, data)

    # Fit polynomial to hydrogen consumption
    i_fit, hydro_fit = fit_polynomial_hydrogen_consumption(df, Hydro_com)

    # Plot hydrogen consumption
    plt.figure(figsize=(6, 6))
    plt.plot(df['i (A/cm²)'], Hydro_com, 'bo', label='Hydrogen Consumption Data')
    plt.plot(i_fit, hydro_fit, 'b-', label='Hydrogen Consumption (fitted)')
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Hydrogen Consumption (g/s)')
    plt.legend()
    plt.savefig('Hydrogen_Consumption.png', dpi=200)
    plt.show()

    # Calculate WLTC hydrogen consumption
    time, speed, interpolated_current_density = calculate_wltc_hydrogen_consumption('WLTC.xlsx', df)

    # More code following to calculate total hydrogen consumption and efficiency...

if __name__ == "__main__":
    main()
