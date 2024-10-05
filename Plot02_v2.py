import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Plot01_V1 import calculate_forces, calculate_power, simulate_soc_and_power

#-------------------- Parameter Initialization ----------------------------#
file = 'Plot02.xlsx'
df = pd.read_excel(file, header=0)

# Fuel cell system parameters
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

#-------------------- Question a 1: Compressor Power Calculation ----------------------------#
P_com = np.zeros(len(df))

for i in range(len(df)):
    P_com[i] = (1 / data['Air_compressor_efficiency'][0]) * \
                (data['Air_stoich'][0] / data['O2_molar_fraction_ambient_air'][0]) * \
                (data['N_cell'][0] * df['i (A/cm²)'][i]) / (4 * data['Faraday_const'][0]) * \
                (data['Gamma'][0] / (data['Gamma'][0] - 1)) * \
                (data['R'][0] * data['Ambient_temperature'][0]) * \
                ((df['P air (bar)'][i] / data['P_atm'][0])**((data['Gamma'][0] - 1) / data['Gamma'][0]) - 1) * data['Area_stack'][0] / 1000

# Plot compressor power
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.plot(df['U_cell (V)'], P_com, 'y-.', label='Power of Air Compressor')
plt.title('Power of Air Compressor')
plt.xlabel('Voltage (V)')
plt.ylabel('Compressor Power (kW)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(P_com, 'g-', label='Power of Air Compressor')
plt.xlabel('Index')
plt.ylabel('Compressor Power (kW)')
plt.legend()

plt.tight_layout()
plt.savefig('Power_of_Air_compressor.png', dpi=200)
plt.show()

#-------------------- Question a 2: Fuel Cell Power Calculation ----------------------------#
P_fuel = np.zeros(len(df))

for j in range(len(df)):
    P_fuel[j] = df['U_cell (V)'][j] * df['i (A/cm²)'][j] * data['N_cell'][0] * data['Area_stack'][0]

# Plot fuel cell power
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.plot(df['U_cell (V)'], P_fuel / 1000, 'b-.', label='Power by Fuel Cell')
plt.title('Power by Fuel Cell')
plt.xlabel('Voltage (V)')
plt.ylabel('Power (kW)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(P_fuel / 1000, 'r-', label='Power by Fuel Cell')
plt.xlabel('Index')
plt.ylabel('Power (kW)')
plt.legend()

plt.tight_layout()
plt.savefig('Power_of_Fuel_Cell.png', dpi=200)
plt.show()

#-------------------- Hydrogen Consumption Calculation ----------------------------#
Hydro_com = np.zeros(len(df))

for i in range(len(df)):
    molar_flow = df['i (A/cm²)'][i] / (2 * data['Faraday_const'][0])
    Hydro_com[i] = molar_flow * data['Molar_mass_dihydrogen'][0] * data['N_cell'][0] * data['Area_stack'][0]

# Fit a 5th-degree polynomial to hydrogen consumption data
coefficients = np.polyfit(df['i (A/cm²)'], Hydro_com, 5)
polynomial = np.poly1d(coefficients)

# Generate fitted values for plotting
i_fit = np.linspace(min(df['i (A/cm²)']), max(df['i (A/cm²)']), 1000)
hydro_fit = polynomial(i_fit)

# Plot hydrogen consumption vs current density
plt.figure(figsize=(6, 6))
plt.plot(df['i (A/cm²)'], Hydro_com, 'bo', label='Hydrogen Consumption Data')
plt.plot(i_fit, hydro_fit, 'r-', label='5th Degree Polynomial Fit')
plt.title('Hydrogen Consumption vs Current Density')
plt.xlabel('Current Density (A/cm²)')
plt.ylabel('Hydrogen Consumption (g)')
plt.legend()
plt.show()

#-------------------- Drive Cycle Hydrogen Consumption Calculation ----------------------------#
WLTC = pd.read_excel('Plot01.xlsx', sheet_name='Sheet1', header=0)
speed = WLTC.iloc[:, 1].values
time = WLTC.iloc[:, 0].values

# Interpolate current density to match WLTC time steps
current_density = df['i (A/cm²)'].values
interp_function = interp1d(np.linspace(0, time[-1], len(current_density)), current_density, kind='linear', fill_value='extrapolate')
interpolated_current_density = interp_function(time)

# Forces and power calculation based on speed
v_s, acceleration, Fair, Frolling, Fcl = calculate_forces(time, speed)
Instant_power, InP_Hybrid, Bat_motor_gen, Bat_motor_demand = calculate_power(time, v_s, acceleration, Fair, Frolling, Fcl)
SoC, power_battery, Power_demand, power_hybrid = simulate_soc_and_power(time, InP_Hybrid)

#-------------------- WLTC and Polarization Curve Analysis ----------------------------#
# Load the WLTC and polarization curve data
wltc_data = pd.read_excel('Plot01.xlsx', header=0)  # Ensure the path is correct
polarization_curve = df  # Ensure df is defined correctly

# Define parameters
A_cell = data['Area_stack'].values[0]  # cm², ensure data is defined
N_cells = data['N_cell'].values[0]  # Ensure data is defined
threshold = 0.0001  # Convergence threshold for voltage

# Initialize results
results = []

# Ensure time and Power_demand are defined
for t in range (len(time)):
    P_demand = Power_demand[t]  # Power demand at time t
    U_cell = 0.85  # Initial guess for voltage
    iteration = 0

    while True:
        # Calculate current
        I_stack = P_demand / U_cell if U_cell != 0 else 0  # Prevent division by zero
        # Calculate current density
        i_t = I_stack / A_cell if A_cell != 0 else 0  # Prevent division by zero
        # Find the closest matching voltage from the polarization curve
        closest_index = (polarization_curve['i (A/cm²)'] - i_t).abs().idxmin()  # Get index of the closest value
        U_cell_new = polarization_curve.loc[closest_index, 'U_cell (V)']

        # Check for convergence
        if abs(U_cell_new - U_cell) < threshold or iteration > 100:
            break
        
        U_cell = U_cell_new
        iteration += 1

    results.append({'Time': t, 'P_demand': P_demand, 'I_stack': I_stack, 'i_t': i_t, 'U_cell': U_cell})

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)
print(results_df)