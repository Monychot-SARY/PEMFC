import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from Plot01_V1 import calculate_forces, calculate_power

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

# Interpolate current density to match WLTC time steps
current_density = df['i (A/cm²)'].values
interp_function = interp1d(np.linspace(0, time[-1], len(current_density)), current_density, kind='linear', fill_value='extrapolate')
interpolated_current_density = interp_function(time)

# Forces and power calculation based on speed
v_s, acceleration, Fair, Frolling, Fcl = calculate_forces(time, speed)
Instant_power, Hybrid, Bat_motor_gen, Bat_motor_demand = calculate_power(time, v_s, acceleration, Fair, Frolling, Fcl)

# Hydrogen consumption during drive cycle
Hydro_com_interp = np.zeros(len(time))
for i in range(len(time)):
    molar_flow = interpolated_current_density[i] / (2 * data['Faraday_const'][0])
    Hydro_com_interp[i] = molar_flow * data['Molar_mass_dihydrogen'][0] * data['N_cell'][0] * data['Area_stack'][0] / 1000  # kg
#-------------------- Total Hydrogen Consumption and Range ----------------------------#
total_hydrogen_consumed = np.sum(Hydro_com_interp * np.diff(time, prepend=0))

total_distance_m = np.sum(speed * np.diff(time, prepend=0))  # Total distance in meters
total_distance_km = total_distance_m / 1000  # Convert to km
average_hydro_com_per_100km = (total_hydrogen_consumed / total_distance_km) * 100  # kg H2 per 100 km
operating_range = data['M_tank'] / (total_hydrogen_consumed / total_distance_km)  # in km

#-------------------- Average Energetic Efficiency ----------------------------#
LHV_H2 = 120_000  # Lower heating value of H2 in kJ/kg
total_energy_content = total_hydrogen_consumed * LHV_H2  # Total energy content in kJ

avg_power_output = np.mean(P_fuel) / 1000  # in kW
total_time_hours = np.sum(np.diff(time, prepend=0)) / 3600  # Convert to hours

efficiency = (avg_power_output * total_time_hours) / (total_energy_content / 3600)  # in %

#-------------------- Print Final Results ----------------------------#
# Extract scalar value for the operating range if it's a Series
if isinstance(operating_range, pd.Series):
    operating_range = operating_range.iloc[0]

# Extract scalar value for average_hydro_com_per_100km if necessary
if isinstance(average_hydro_com_per_100km, pd.Series):
    average_hydro_com_per_100km = average_hydro_com_per_100km.iloc[0]

# Extract scalar value for total_hydrogen_consumed if necessary
if isinstance(total_hydrogen_consumed, pd.Series):
    total_hydrogen_consumed = total_hydrogen_consumed.iloc[0]

#-------------------- Print Final Results ----------------------------#
print(f"Vehicle Operating Range: {operating_range:.2f} km")
print(f"Average Hydrogen Consumption: {average_hydro_com_per_100km:.4f} kgH2/100 km")
print(f"Average Fuel Cell Efficiency: {efficiency:.2f}%")
print(f"Total Hydrogen Consumed: {total_hydrogen_consumed:.4f} kg")



# Plot interpolated hydrogen consumption
plt.plot(time, Hydro_com_interp, label='Interpolated Hydrogen Consumption (kg/s)')
plt.xlabel('Time (s)')
plt.ylabel('Hydrogen Consumption (kg/s)')
plt.legend()
plt.show()

