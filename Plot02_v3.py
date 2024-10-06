import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

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

# Main function to execute all steps
def main():
    # Load data
    data, df = load_data('Plot02.xlsx')

    # Calculate compressor power
    P_com = calculate_compressor_power(df, data)

    # Plot compressor power
    plt.figure(figsize=(6, 6))
    plt.plot(df['U_cell (V)'], P_com, 'y-.', label='Power of Air Compressor')
    plt.xlabel('Cell Voltage (V)', fontsize=12)
    plt.ylabel('Compressor Power (kW)', fontsize=12)
    plt.title('Compressor Power vs Cell Voltage', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig('Power_of_Air_compressor.png', dpi=200)
    plt.show()

    # Calculate fuel cell power
    P_fuel = calculate_fuel_cell_power(df, data)

    # Plot fuel cell power
    plt.figure(figsize=(6, 6))
    plt.plot(df['U_cell (V)'], P_fuel / 1000, 'b-.', label='Fuel Cell Power')
    plt.xlabel('Cell Voltage (V)', fontsize=12)
    plt.ylabel('Fuel Cell Power (kW)', fontsize=12)
    plt.title('Fuel Cell Power vs Cell Voltage', fontsize=14)
    plt.grid(True)
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
    plt.plot(i_fit, hydro_fit, 'r-', label='Hydrogen Consumption (fitted)')
    plt.xlabel('Current Density (A/cm²)', fontsize=12)
    plt.ylabel('Hydrogen Consumption (g/s)', fontsize=12)
    plt.title('Hydrogen Consumption vs Current Density', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig('Hydrogen_Consumption.png', dpi=200)
    plt.show()

    # Calculate WLTC hydrogen consumption
    time, speed, interpolated_current_density = calculate_wltc_hydrogen_consumption('WLTC.xlsx', df)

    # Additional processing for WLTC and hydrogen consumption analysis can go here

if __name__ == "__main__":
    main()
