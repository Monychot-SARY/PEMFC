import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
rho = 1.2  # Air density (kg/m^3)
Cd = 0.29  # Drag coefficient
A = 2.25  # Vehicle cross-sectional area (m^2)
mass = 2000  # Vehicle mass (kg)
Cr = 0.0115  # Rolling resistance coefficient
g = 9.81  # Gravitational acceleration (m/s^2)
alpha = 0  # Incline angle (rad)
converter_efficiency = 0.9
aux_output_power = 300  # in Watts
battery_capacity = 1.24  # Battery capacity in kWh
SoC_min = 50  # Minimum SoC percentage
SoC_max = 65  # Maximum SoC percentage
discharge_power_battery = 12.4   # Discharge power in Watts
charge_power_battery_10C = 12.4   # Charge power for SoC < 55% (10C)
charge_power_battery_6C = 7.44   # Charge power for SoC > 55% (6C)
fuel_cell_min_power = 2.5  # Minimum fuel cell power in kW

# Function to read and filter data
def read_data(file_name, sheet_name='Sheet1', time_limit=1800):
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        t = df.iloc[:, 0]
        v = df.iloc[:, 1]
        mask = (t >= 0) & (t <= time_limit)
        return t[mask].values, v[mask].values  # Ensure to return as numpy arrays
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def calculate_forces(t, v):
    v_s = v * 1000 / 3600  # Convert speed to m/s
    delta_v = np.diff(v_s)
    delta_t = np.diff(t)

    acceleration = np.zeros_like(v_s)
    acceleration[1:] = delta_v / delta_t  # Calculate acceleration

    # Forces calculations must have the same length as t
    Fair = 0.5 * rho * v_s**2 * Cd * A  # Air resistance
    Frolling = np.ones_like(v_s) * (mass * Cr * g * np.cos(alpha))  # Rolling resistance
    Fcl = np.ones_like(v_s) * (mass * g * np.sin(alpha))  # Climbing resistance

    return v_s, acceleration, Fair, Frolling, Fcl

# Function to calculate total force and power
def calculate_power(t, v_s, acceleration, Fair, Frolling, Fcl):
    F_total = Fcl + Frolling + Fair + mass * acceleration  # Total force
    InP = v_s * F_total  # in Watts
    InP[0] = 0  # Set initial power to zero

    # Initialize arrays
    Pgen = np.zeros(len(t))
    Pdemand = np.zeros(len(t))
    Bat_motor_gen = np.zeros(len(t))
    Bat_motor_demand = np.zeros(len(t))
    InP_Hybrid = np.zeros(len(t))
    Bat = np.ones(len(t)) * aux_output_power / converter_efficiency

    for i in range(len(t)):
        if InP[i] <= 0:
            Pgen[i] = InP[i]
            Bat_motor_gen[i] = Pgen[i] * converter_efficiency
        else:
            Pdemand[i] = InP[i]
            Bat_motor_demand[i] = Pdemand[i] / converter_efficiency  # in Watts

    for i in range(len(t)):
        InP_Hybrid[i] = Bat[i] + Bat_motor_gen[i] + Bat_motor_demand[i]  # in Watts
        
    return InP, InP_Hybrid, Bat_motor_gen, Bat_motor_demand

# Function to simulate battery SoC and fuel cell dynamics
def simulate_soc_and_power(t, InP_Hybrid):
    SoC = np.zeros(len(t))
    SoC[0] = 60  # Start with maximum SoC

    power_battery = np.zeros(len(t))
    power_fuel_cell = np.zeros(len(t))
    power_hybrid = np.zeros(len(t))
    power_fuel_cell[0] = fuel_cell_min_power
    # Power demand
    power_demand = InP_Hybrid / 1000  # Convert Watts to kW

    # Simulation loop for both charging and discharging
    for i in range(1, len(t)):
        if power_demand[i] > 0:  # Battery discharging
            if SoC[i - 1] < 55:  # SoC < 55% --> 10C discharge rate
                power_battery[i] = 0.05 * power_demand[i]  # 5% of total demand
                charging_power = charge_power_battery_10C
            else:  # SoC > 55% --> 6C discharge rate
                power_battery[i] = 0.30 * power_demand[i]  # 30% of total demand
                charging_power = charge_power_battery_6C

            # Cap battery discharge power
            power_battery[i] = min(power_battery[i], discharge_power_battery)

            # Fuel cell power adjustment
            if power_demand[i] - power_battery[i] <= fuel_cell_min_power:
                power_fuel_cell[i] = fuel_cell_min_power
            else: 
                power_fuel_cell[i] = power_demand[i] - power_battery[i]

        elif power_demand[i] < 0:  # Battery charging
            charging_power = -power_demand[i]  # Make it positive

            if SoC[i - 1] < 55:  # SoC < 55% --> 10C charging rate
                max_charging_power = charge_power_battery_10C
            else:  # SoC > 55% --> 6C charging rate
                max_charging_power = charge_power_battery_6C
            # Cap the charging power
            charging_power = min(charging_power, max_charging_power)

            power_battery[i] = -charging_power  # Charging, negative power

        # Update SoC based on your formula
        SoC[i] = min(SoC_max, max(SoC_min, SoC[i - 1] - ((power_battery[i] * 1 / 3600) / battery_capacity) +
                        ((-power_demand[i] * 1 / 3600) / battery_capacity)))

        # Ensure SoC remains within bounds
        if SoC[i] < SoC_min:
            SoC[i] = SoC_min  # Cap at minimum SoC
            power_battery[i] = 0  # Prevent further discharge
        elif SoC[i] > SoC_max:
            SoC[i] = SoC_max  # Cap at maximum SoC
            power_battery[i] = 0  # Prevent further charging
            
        power_hybrid[i] = power_battery[i] + power_fuel_cell[i]

    return SoC, power_battery, power_fuel_cell, power_hybrid

# Function to plot results
def plot_results(t, v, v_s, acceleration, Fair, Frolling, Fcl, InP, SoC, power_battery, power_fuel_cell, power_hybrid):
    plt.figure(figsize=(10, 15))
    
    # Plot speed and forces
    plt.subplot(5, 1, 1)
    plt.plot(t, v, 'r-', label='Speed (km/h)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.xlim([t[0], t[-1]])
    plt.legend()
    plt.annotate(f'Min: {min(v):.2f} km/h', xy=(t[np.argmin(v)], min(v)), xytext=(t[np.argmin(v)], min(v) + 5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8, color='black')
    plt.annotate(f'Max: {max(v):.2f} km/h', xy=(t[np.argmax(v)], max(v)), xytext=(t[np.argmax(v)], max(v) - 5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=8, color='black')

    plt.subplot(5, 1, 2)
    plt.plot(t, acceleration, 'g-', label='Acceleration (m/s^2)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.xlim([t[0], t[-1]])
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(t, Fair, label='Air Resistance (N)', color='orange')
    plt.plot(t, Frolling, label='Rolling Resistance (N)', color='purple')
    plt.plot(t, Fcl, label='Climbing Resistance (N)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Forces (N)')
    plt.legend()
    
    # Plot power and SoC
    plt.subplot(5, 1, 4)
    plt.plot(t, power_hybrid, label='Power Hybrid (kW)', color='teal')
    plt.plot(t, power_battery, label='Battery Power (kW)', color='red')
    plt.plot(t, power_fuel_cell, label='Fuel Cell Power (kW)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (kW)')
    plt.legend()
    
    plt.subplot(5, 1, 5)
    plt.plot(t, SoC, label='State of Charge (%)', color='green')
    plt.axhline(y=SoC_min, color='red', linestyle='--', label='Min SoC')
    plt.axhline(y=SoC_max, color='orange', linestyle='--', label='Max SoC')
    plt.xlabel('Time (s)')
    plt.ylabel('State of Charge (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Question_C.png')
    plt.show()

# Main function to execute the simulation
def main():
    t, v = read_data('Plot01_130.xlsx')
    if t is None or v is None:
        return

    v_s, acceleration, Fair, Frolling, Fcl = calculate_forces(t, v)
    InP, InP_Hybrid, Bat_motor_gen, Bat_motor_demand = calculate_power(t, v_s, acceleration, Fair, Frolling, Fcl)
    SoC, power_battery, power_fuel_cell, power_hybrid = simulate_soc_and_power(t, InP_Hybrid)
    plot_results(t, v, v_s, acceleration, Fair, Frolling, Fcl, InP, SoC, power_battery, power_fuel_cell, power_hybrid)

# Run the simulation
if __name__ == '__main__':
    main()
