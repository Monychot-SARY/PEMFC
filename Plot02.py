import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#-------------------- Parameter ----------------------------#
file = 'Plot02.xlsx'
df = pd.read_excel(file, header=0)

data = pd.DataFrame({
    "M_H2": [5.6],  # kg
    "P_H2": [700],  # bar
    "Molar_mass_dihydrogen": [2.016],  # g/mol
    "LHV": [242],  # kJ/mol
    "HHV": [285],  # kJ/mol
    "H2_loss": [2],  # %
    "Faraday_const": [96487],  # C/mol
    "Air_stoich": [1.5],
    "P_atm": [1],  # bar
    "O2_molar_fraction_ambient_air": [21/100],  # %
    "Air_compressor_efficiency": [60/100],  # %
    "N_cell": [330],
    "Area_stack": [273],  # cm^2
    "Gamma": [1.4],
    "R": [8.314],  # J/mol.K (Ideal gas constant)
    "Ambient_temperature": [298]  # K
})


#-------------------- Question a 1 ----------------------------#

tor = np.zeros(len(df.iloc[0:,]))
P_com = np.zeros(len(df.iloc[0:,]))

for i in range(len(df.iloc[0:,])):
    P_com[i] = (1 / data['Air_compressor_efficiency'][0]) * \
               (data['Air_stoich'][0] / data['O2_molar_fraction_ambient_air'][0]) * \
               (data['N_cell'][0] * df['i (A/cm²)'][i]) / (4 * data['Faraday_const'][0]) * \
               (data['Gamma'][0] / (data['Gamma'][0] - 1)) * \
               (data['R'][0] * data['Ambient_temperature'][0]) * \
               ((df['P air (bar)'][i]/data['P_atm'][0])** ((data['Gamma'][0]-1) / (data['Gamma'][0]))-1)*data['Area_stack'][0]/1000
aa = (1 / data['Air_compressor_efficiency'][0])
# Plot the compressor power (P_com)
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.plot(df['U_cell (V)'], P_com, 'y-.', label='Power of Air Compressor')  # Use range(len(df)) for the x-axis
plt.title('Power of Air Compressor')
plt.xlim([min(df['U_cell (V)']), 1*max(df['U_cell (V)'])])
plt.xlabel('Voltage (V)')
plt.ylabel('Compressor Power (kW)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot( P_com, 'g-', label='Power of Air Compressor')  # Use range(len(df)) for the x-axis
# plt.title('Power of Air Compressor')
plt.xlim([0, len(df.iloc[0:,])])
plt.xlabel('')
plt.ylabel('Compressor Power (kW)')
plt.legend()

plt.tight_layout()
plt.savefig('Power_of_Air_compressor.png', dpi=200)
plt.show()


#-------------------- Question a 2 ----------------------------#

P_fuel = np.zeros(len(df.iloc[0:,]))

for j in range(len(df.iloc[0:,])):
    P_fuel[j] =  df['U_cell (V)'][j]*df['i (A/cm²)'][j]*data['N_cell'][0]*data['Area_stack'][0]

# Plot the compressor power (P_com)
plt.figure(figsize=(6, 6))
plt.subplot(2, 1, 1)
plt.plot(df['U_cell (V)'], P_fuel/1000, 'b-.', label='Power Electricity by Fuel cell')  # Use range(len(df)) for the x-axis
plt.title('Power Electricity by Fuel cell')
plt.xlim([min(df['U_cell (V)']), 1*max(df['U_cell (V)'])])
plt.xlabel('Voltage (V)')
plt.ylabel('Power Electricity by Fuel cell (kW)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot( P_fuel/1000, 'r-', label='Power Electricity by Fuel cell')  # Use range(len(df)) for the x-axis
plt.xlim([0, len(df.iloc[0:,])])
plt.xlabel('')
plt.ylabel('Power Electricity by Fuel cell (kW)')
plt.legend()

plt.tight_layout()
plt.savefig('Power_of_Fuel_Cell.png', dpi=200)
plt.show()

#-------------------- Question a 3 ----------------------------#

Hydro_com = np.zeros(len(df.iloc[0:,]))
molor_flow = np.zeros(len(df.iloc[0:,]))
for i in range( len(df.iloc[0:,])):
    molor_flow[i] = df['i (A/cm²)'][i]/(2*data['Faraday_const'][0])
    
plt.plot(molor_flow, 'r-')  # Use range(len(df)) for the x-axis
plt.xlim([0, len(df.iloc[0:,])])
plt.xlabel('')
plt.ylabel('')
plt.show()