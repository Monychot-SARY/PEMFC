import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Plot01_V1 import calculate_forces, calculate_power, simulate_soc_and_power
from scipy.optimize import minimize_scalar
import os
#-------------------- Parameter Initialization ----------------------------#
plt.close('all')
os.system('cls')  # For Windows
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
SoC, power_battery,power_fuel_cell, power_hybrid = simulate_soc_and_power(time, InP_Hybrid)

#-------------------- WLTC and Polarization Curve Analysis ----------------------------#
# Load the WLTC and polarization curve data
wltc_data = pd.read_excel('Plot01.xlsx', header=0)  # Ensure the path is correct
polarization_curve = df  # Ensure df is defined correctly

# Define parameters
A_cell = data['Area_stack']
N_cells = data['N_cell']

# Load the WLTC and polarization curve data
wltc_data = pd.read_excel('Plot01.xlsx', header=0)  # Ensure the path is correct
polarization_curve = df  # Ensure df is defined correctly

# Define parameters
A_cell = data['Area_stack']
N_cells = data['N_cell']

# Function to create an interpolation function for the polarization curve
def create_interpolation(polarization_curve):
    return interp1d(polarization_curve['i (A/cm²)'], polarization_curve['U_cell (V)'], fill_value="extrapolate")

# Function to calculate current
def calculate_current(P_demand, U_cell_value):
    if isinstance(U_cell_value, (pd.Series, pd.DataFrame)):
        U_cell_value = U_cell_value.item()  # Ensure U_cell_value is a scalar
    return P_demand / U_cell_value if U_cell_value != 0 else 0

# Function to calculate current density
def calculate_current_density(I_stack, A_cell):
    if isinstance(A_cell, (pd.Series, pd.DataFrame)):
        A_cell = A_cell.item()  # Ensure A_cell is a scalar
    return I_stack / A_cell if A_cell != 0 else 0

def compute_results_optimization(time, Power_demand, A_cell, polarization_curve, N_cells, lb, ub, threshold):
    U_cell = 0.6
    results = []

    # Create interpolation function outside the loop
    interpolation_function = create_interpolation(polarization_curve)

    # Ensure lb and ub are scalars
    lb = lb.item() if isinstance(lb, (pd.Series, pd.DataFrame)) else lb
    ub = ub.item() if isinstance(ub, (pd.Series, pd.DataFrame)) else ub

    for t in range(len(time)):
        P_demand = Power_demand[t]
        iteration = 0

        while True:
            # Define the function to minimize with captured parameters
            def func(U_cell_guess):
                I_stack = calculate_current(P_demand, U_cell_guess)  # Update current with new guess
                i_t = calculate_current_density(I_stack, A_cell)  # Calculate current density
                interpolated_value = interpolation_function(i_t.item())
                result = (U_cell_guess - interpolated_value.item()) **2 # Squared difference for minimization
                # print(f"func({U_cell_guess}) = {result}")  # Debugging line to see the output
                return result
            
            # Perform optimization to find the root
            res = minimize_scalar(func, bounds=(lb, ub), method='bounded')

            if res.success:
                U_cell_new = res.x  # This should be a scalar

                # Check for convergence
                if abs(U_cell_new - U_cell) < threshold or iteration > 1000:
                    # print("Converged or exceeded iterations.")
                    break
                
                U_cell = U_cell_new  # Update U_cell for the next iteration
                iteration += 1
            else:
                print("Optimization failed.")
                break

        # Calculate final values after convergence
        I_stack = calculate_current(P_demand, U_cell)
        i_t = calculate_current_density(I_stack, A_cell)

        results.append({
            'Time': t,
            'P_demand': P_demand,
            'I_stack': I_stack,
            'i_t': i_t,
            'U_cell': U_cell
        })

    return results
lb = 0  # Lower bound for U_cell
ub = 0.85 # Upper bound for U_cell
error =10e-10
Result_optimization = compute_results_optimization(time, InP_Hybrid, A_cell, polarization_curve, N_cells, lb, ub, error)
Result_optimization = pd.DataFrame(Result_optimization)
I_cell = np.zeros(len(time))

mass_hydro = np.zeros(len(time))
for i in range(len(time)):
    P_demand = Result_optimization['P_demand'][i]  # Power demand at each time step
    U_cell = Result_optimization['U_cell'][i]      # Optimized cell voltage
    if P_demand < 0: 
        mass_hydro[i] = 0
    else: 
        if U_cell > 0:
            I_cell[i] = P_demand / (U_cell * data['N_cell'][0])  # Calculate current
        else:
            I_cell[i] = 0  # Avoid division by zero if U_cell is 0
        # Calculate hydrogen mass in grams and convert to kilograms
        mass_hydro[i] = ((I_cell[i] * data['N_cell'][0] * data['Molar_mass_dihydrogen'][0]) / (2 * data['Faraday_const'][0])) / 1000  # Convert to kg

# Calculate total hydrogen consumed in kg
total_hydrogen_consumed = np.sum(mass_hydro)  # Total in kg
# Constants (LHV for hydrogen, etc.)
LHV_hydrogen = data['LHV'][0] * 1000  # Convert from kJ/mol to J/mol
Molar_mass_h2 = data['Molar_mass_dihydrogen'][0] / 1000  # Convert to kg/mol

# System efficiency calculation over time
efficiency = np.zeros(len(time))

for i in range(len(time)):
    P_fuel_cell = Result_optimization['P_demand'][i]  # Power demand or power from the fuel cell
    if mass_hydro[i] > 0:
        # Energy from hydrogen (LHV * mass flow rate of H2)
        energy_from_h2 = mass_hydro[i] * LHV_hydrogen / Molar_mass_h2
        # System efficiency at each time step
        efficiency[i] = P_fuel_cell / energy_from_h2 if energy_from_h2 > 0 else 0
    else:
        efficiency[i] = efficiency[i-1]  # Avoid division by zero

# Calculate average efficiency over the entire cycle
average_efficiency = np.mean(efficiency) * 100  # Convert to percentage

distance_traveled = v_s[1:] * np.diff(time)
total_distance = np.sum(distance_traveled)/1000
# Calculate average hydrogen consumption in kg(H2)/100 km
if total_distance > 0:
    average_hydrogen_consumption = (total_hydrogen_consumed * 100) / total_distance
else:
    average_hydrogen_consumption = 0
hydrogen_capacity = data['M_tank'][0]  # Hydrogen tank capacity in kg
operation_range = hydrogen_capacity / (average_hydrogen_consumption / 100) if average_hydrogen_consumption > 0 else 0  # Range in km
# Print results
print(f'Total hydrogen consumed: {total_hydrogen_consumed:.6f} kg')
print(f'Average System Efficiency: {average_efficiency:.2f}%')
print(f'Average Hydrogen Consumption: {average_hydrogen_consumption:.6f} kg(H2)/km')
print(f'Operation Range: {operation_range:.2f} km')
# Create a figure with two subplots
plt.figure(figsize=(10, 8))
# Plot hydrogen mass consumption over time (subplot 1)
plt.subplot(2, 1, 1)
plt.plot(time, mass_hydro, 'b-', label='Hydrogen Mass (kg/s)')
plt.title(f'Hydrogen Mass Consumption over Time\nTotal hydrogen consumed: {total_hydrogen_consumed:.6f} kg', fontsize=12)
plt.xlabel('Time (s)')
plt.ylabel('Hydrogen Mass (kg/s)')
plt.ylim([min(mass_hydro),max(mass_hydro)*1.1])
plt.xlim([time[0],len(time)])
plt.legend()
# Plot system efficiency over time (subplot 2)
plt.subplot(2, 1, 2)
plt.plot(time, efficiency * 100, 'm-', label='System Efficiency (%)')
plt.title(f'System Efficiency over Time\nAverage System Efficiency: {average_efficiency:.2f}%', fontsize=12)
plt.xlabel('Time (s)')
plt.ylabel('Efficiency (%)')
plt.xlim([time[0],len(time)])
plt.ylim([min(efficiency * 100), max(efficiency * 100) * 1.1])  # Corrected here
plt.legend()

plt.tight_layout()  # Adjust the rect parameter to leave space for the suptitle
plt.savefig('Hydrogen_Mass_and_Efficiency.png', dpi=200)
plt.show()
