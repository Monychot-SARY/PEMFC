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
        # Filter data by time limit
        mask = (t >= 0) & (t <= time_limit)
        return t[mask], v[mask]
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Function to calculate forces and acceleration
def calculate_forces(t, v):
    v_s = v * 1000 / 3600  # Convert speed to m/s
    delta_v = np.diff(v_s)
    delta_t = np.diff(t)
    acceleration = np.zeros_like(v_s)
    acceleration[1:] = delta_v / delta_t  # Adjust length by 1

    Fair = 0.5 * rho * v_s**2 * Cd * A  # Air resistance
    Frolling = mass * Cr * g * np.cos(alpha)  # Rolling resistance (constant)
    Fcl = mass * g * np.sin(alpha)  # Climbing resistance (constant)

    return v_s, acceleration, Fair, Frolling, Fcl

# Function to calculate total force and power
def calculate_power(t, v_s, acceleration, Fair, Frolling, Fcl):
    F_total = Fcl + Frolling + Fair[1:] + mass * acceleration[1:]  # Total force
    InP = v_s[1:] * F_total  # Instant power in Watts
    InP = np.insert(InP, 0, 0)  # Set first value to zero

    # DC/DC Converter and battery power handling
    Pgen, Pdemand, Bat_motor_gen, Bat_motor_demand, InP_Hybrid = np.zeros((5, len(t)))
    Bat = aux_output_power / converter_efficiency

    for i in range(1, len(t)):
        if InP[i] <= 0:
            Pgen[i] = InP[i]
            Bat_motor_gen[i] = Pgen[i] * converter_efficiency
        else:
            Pdemand[i] = InP[i]
            Bat_motor_demand[i] = Pdemand[i] / converter_efficiency

        InP_Hybrid[i] = Bat + Bat_motor_gen[i] + Bat_motor_demand[i]  # Hybrid power

    return InP, InP_Hybrid, Bat_motor_gen, Bat_motor_demand

# Function to simulate battery SoC and fuel cell dynamics
def simulate_soc_and_power(t, InP):
    SoC = np.zeros(len(t))
    SoC[0] = 60  # Start at 60% SoC
    power_battery, power_fuel_cell, power_hybrid = np.zeros((3, len(t)))
    power_fuel_cell[0] = fuel_cell_min_power
    power_demand = InP / 1000  # Convert to kW

    for i in range(1, len(t)):
        charging_power = 0

        if power_demand[i] > 0:  # Battery discharging
            if SoC[i - 1] < 55:
                power_battery[i] = 0.05 * power_demand[i]
            else:
                power_battery[i] = 0.30 * power_demand[i]

            power_battery[i] = min(power_battery[i], discharge_power_battery)
            power_fuel_cell[i] = power_demand[i] - power_battery[i]
            if power_fuel_cell[i] < fuel_cell_min_power:
                power_fuel_cell[i] = fuel_cell_min_power

        else:  # Battery charging
            charging_power = -power_demand[i]
            max_charging_power = charge_power_battery_10C if SoC[i - 1] < 55 else charge_power_battery_6C
            power_battery[i] = -min(charging_power, max_charging_power)

        # # Update SoC
        # SoC[i] = np.clip(SoC[i - 1] - (power_battery[i] / battery_capacity) * (1 / 3600), SoC_min, SoC_max)
        SoC[i] = min(SoC_max, max(SoC_min, SoC[i - 1] - ((power_battery[i] * 1 / 3600) / battery_capacity) +
                (( -power_demand[i] * 1 / 3600) / battery_capacity)))
        power_hybrid[i] = power_battery[i] + power_fuel_cell[i]

    return SoC, power_battery, power_fuel_cell, power_hybrid

# Function to plot results
def plot_results(t, v, v_s, acceleration, Fair, Frolling, Fcl, InP, SoC, power_battery, power_fuel_cell, power_hybrid):
    plt.figure(figsize=(10, 15))
    time = len(t)
    # Plot speed and forces
    plt.subplot(5, 1, 1)
    plt.plot(t, v, 'r-', label='Speed (km/h)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(v):.2f} km/h', xy=(t[np.argmin(v)], min(v)), xytext=(t[np.argmin(v)], min(v) + 5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(v):.2f} km/h', xy=(t[np.argmax(v)], max(v)), xytext=(t[np.argmax(v)], max(v) - 5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(5, 1, 2)
    plt.plot(t, v_s, 'b-', label='Speed (m/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(v_s):.2f} m/s', xy=(t[np.argmin(v_s)], min(v_s)), xytext=(t[np.argmin(v_s)], min(v_s) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(v_s):.2f} m/s', xy=(t[np.argmax(v_s)], max(v_s)), xytext=(t[np.argmax(v_s)], max(v_s) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(5, 1, 3)
    plt.plot(t[1:], acceleration[1:], 'g-', label='Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(acceleration):.2f} m/s²', xy=(t[1:][np.argmin(acceleration)], min(acceleration)), 
                 xytext=(t[1:][np.argmin(acceleration)], min(acceleration) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(acceleration):.2f} m/s²', xy=(t[1:][np.argmax(acceleration)], max(acceleration)), 
                 xytext=(t[1:][np.argmax(acceleration)], max(acceleration) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(5, 1, 4)
    plt.plot(t, Fair / 1000, 'y-', label='Air Resistance (kN)')
    plt.xlabel('Time (s)')
    plt.ylabel('Air Resistance (kN)')
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(Fair / 1000):.2f} kN', xy=(t[np.argmin(Fair)], min(Fair / 1000)), 
                 xytext=(t[np.argmin(Fair)], min(Fair / 1000) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(Fair / 1000):.2f} kN', xy=(t[np.argmax(Fair)], max(Fair / 1000)), 
                 xytext=(t[np.argmax(Fair)], max(Fair / 1000) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(5, 1, 5)
    plt.plot(t, Frolling * np.ones_like(t) / 1000, 'k-', label='Rolling Resistance (kN)')
    plt.xlabel('Time (s)')
    plt.ylabel('Rolling Resistance (kN)')
    plt.xlim([t[0], time])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Plot hybrid power and SoC
    plt.figure(figsize=(10, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, power_hybrid, label="Hybrid Power (kW)")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (kW)")
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(power_hybrid):.2f} kW', xy=(t[np.argmin(power_hybrid)], min(power_hybrid)), 
                 xytext=(t[np.argmin(power_hybrid)], min(power_hybrid) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(power_hybrid):.2f} kW', xy=(t[np.argmax(power_hybrid)], max(power_hybrid)), 
                 xytext=(t[np.argmax(power_hybrid)], max(power_hybrid) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(4, 1, 2)
    plt.plot(t, power_battery, label="Battery Power (kW)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (kW)")
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(power_battery):.2f} kW', xy=(t[np.argmin(power_battery)], min(power_battery)), 
                 xytext=(t[np.argmin(power_battery)], min(power_battery) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(power_battery):.2f} kW', xy=(t[np.argmax(power_battery)], max(power_battery)), 
                 xytext=(t[np.argmax(power_battery)], max(power_battery) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(4, 1, 3)
    plt.plot(t, power_fuel_cell, label="Fuel Cell Power (kW)", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (kW)")
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(power_fuel_cell):.2f} kW', xy=(t[np.argmin(power_fuel_cell)], min(power_fuel_cell)), 
                 xytext=(t[np.argmin(power_fuel_cell)], min(power_fuel_cell) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(power_fuel_cell):.2f} kW', xy=(t[np.argmax(power_fuel_cell)], max(power_fuel_cell)), 
                 xytext=(t[np.argmax(power_fuel_cell)], max(power_fuel_cell) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.subplot(4, 1, 4)
    plt.plot(t, SoC, label="SoC (%)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("SoC (%)")
    plt.xlim([t[0], time])
    plt.legend()
    # Annotate min and max values
    plt.annotate(f'Min: {min(SoC):.2f} %', xy=(t[np.argmin(SoC)], min(SoC)), 
                 xytext=(t[np.argmin(SoC)], min(SoC) + 1),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(SoC):.2f} %', xy=(t[np.argmax(SoC)], max(SoC)), 
                 xytext=(t[np.argmax(SoC)], max(SoC) - 1),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig('Question_C.png', dpi=200)
    plt.show()


# Main execution flow
def main():
    t, v = read_data('Plot01.xlsx')
    if t is None or v is None:
        return  # Exit if no data

    v_s, acceleration, Fair, Frolling, Fcl = calculate_forces(t, v)
    InP, InP_Hybrid, Bat_motor_gen, Bat_motor_demand = calculate_power(t, v_s, acceleration, Fair, Frolling, Fcl)
    SoC, power_battery, power_fuel_cell, power_hybrid = simulate_soc_and_power(t, InP)

    plot_results(t, v, v_s, acceleration, Fair, Frolling, Fcl, InP, SoC, power_battery, power_fuel_cell, power_hybrid)

# Execute the main function
if __name__ == "__main__":
    main()
