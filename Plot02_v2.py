import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from Plot01_V1 import calculate_forces, calculate_power, simulate_soc_and_power
#-------------------- Parameter Initialization ----------------------------#
file = 'Plot02.xlsx'
df = pd.read_excel(file, header=0)

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
P_com = np.zeros(len(df.iloc[0:, ]))

for i in range(len(df.iloc[0:, ])):
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
P_fuel = np.zeros(len(df.iloc[0:, ]))

for j in range(len(df.iloc[0:, ])):
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
Hydro_com = np.zeros(len(df.iloc[0:, ]))

for i in range(len(df.iloc[0:, ])):
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
LHV_H2 = (data['LHV'] / data['Molar_mass_dihydrogen']) *1000# Lower heating value of H2 in kJ/kg

# Interpolate current density to match WLTC time steps
cell_voltage = df['U_cell (V)'].values
interp_function = interp1d(np.linspace(0, time[-1], len(cell_voltage)), cell_voltage, kind='linear', fill_value='extrapolate')
interpolated_cell_voltage = interp_function(time)



# Forces and power calculation based on speed
v_s, acceleration, Fair, Frolling, Fcl = calculate_forces(time, speed)
Instant_power, Hybrid, Bat_motor_gen, Bat_motor_demand = calculate_power(time, v_s, acceleration, Fair, Frolling, Fcl)
SoC, power_battery, power_fuel_cell, power_hybrid = simulate_soc_and_power(time, Instant_power)
hydrogen_consumption_rate = np.zeros(len(time))
delta_H = 286 # the specific enthalpy of hydrogen (typically around 286 kJ/mol for hydrogen fuel cells).
for i in range(len(time)):
    # Instantaneous_current = interpolated_current_density[i] / data['Area_stack']
    hydrogen_consumption_rate[i] = (Instant_power[i]*data['Faraday_const'][0])/(cell_voltage[i]*delta_H*data['N_cell'])
plt.figure()
plt.plot(hydrogen_consumption_rate)
#-------------------- Average Energetic Efficiency ----------------------------#



