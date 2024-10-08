import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constants
rho = 1.2  # Air density (kg/m^3)
Cd = 0.4  # Drag coefficient
A = 2.25  # Vehicle cross-sectional area (m^2)
mass = 2000  # Vehicle mass (kg)
Cr = 0.014  # Rolling resistance coefficient
g = 9.81  # Gravitational acceleration (m/s^2)
alpha = 0  # Incline angle (rad)
converter_efficiency = 0.9
aux_output_power = 300  # in Watts
battery_capacity = 1.86  # Battery capacity in kWh
SoC_min = 50  # Minimum SoC percentage
SoC_max = 65  # Maximum SoC percentage
discharge_power_battery = 12.4   # Discharge power in Watts
charge_power_battery_10C = 24.8   # Charge power for SoC < 55% (10C)
charge_power_battery_6C = 6.2   # Charge power for SoC > 55% (5C)
fuel_cell_min_power = 2.5  # Minimum fuel cell power in kW

# Function to read and filter data
def read_data(file_name, sheet_name='Sheet1', time_limit=1800):
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        t = df.iloc[:, 0]
        v = df.iloc[:, 1]
        mask = (t >= 0) & (t <= 1800)
        t = t[mask]
        v = v[mask]
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
    acceleration[1:] = delta_v / delta_t  # Calculate acceleration

    Fair = 0.5 * rho * v_s**2 * Cd * A  # Air resistance
    Frolling = mass * Cr * g * np.cos(alpha)  # Rolling resistance
    Fcl = mass * g * np.sin(alpha)  # Climbing resistance

    return v_s, acceleration, Fair, Frolling, Fcl

# Function to calculate total force and power
def calculate_power(t, v_s, acceleration, Fair, Frolling, Fcl):
    F_total = Fcl + Frolling + Fair + mass * acceleration  # Total force
    InP = v_s * F_total  # in Watts
    InP[0] = 0

    # DC/DC converter
    converter_efficiency = 0.9
    aux_output_power = 300  # in Watts

    Pgen = np.zeros(len(t))
    Pdemand = np.zeros(len(t))
    Bat_motor_gen = np.zeros(len(t))
    Bat_motor_demand = np.zeros(len(t))
    InP_Hybrid = np.zeros(len(t))
    Bat = np.ones(len(t))*aux_output_power / converter_efficiency

    for i in range(len(t)):
        if InP[i] <= 0:
            Pgen[i] = InP[i]
            Bat_motor_gen[i] = Pgen[i] * converter_efficiency
        elif InP[i] >= 0:
            Pdemand[i] = InP[i]
            Bat_motor_demand[i] = Pdemand[i] / converter_efficiency  # in Watts

    for i in range(len(t)):
        InP_Hybrid[i] = Bat[i] + Bat_motor_gen[i] + Bat_motor_demand[i]  # in Watts
        
    return InP, InP_Hybrid, Bat_motor_gen, Bat_motor_demand,F_total

# Function to simulate battery SoC and fuel cell dynamics
def simulate_soc_and_power(t, InP_Hybrid):
    SoC = np.zeros(len(t))
    SoC[0] = 60  # Start with maximum SoC

    power_battery = np.zeros(len(t))
    power_fuel_cell = np.zeros(len(t))
    power_hybrid = np.zeros(len(t))
    power_fuel_cell[0] = fuel_cell_min_power
    # Power demand
    power_demand = InP_Hybrid/1000

    # Simulation loop for both charging and discharging
    for i in range(1,len(t)):
        if power_demand[i] > 0:  # Battery discharging
            if SoC[i - 1] < 55:  # SoC < 55% --> 10C discharge rate
                power_battery[i] = 0.05 * power_demand[i]  # 5% of total demand
                charging_power = charge_power_battery_10C
            elif SoC[i - 1] >= 55:  # SoC > 55% --> 6C discharge rate
                power_battery[i] = 0.30 * power_demand[i]  # 30% of total demand
                charging_power = charge_power_battery_6C

            # Cap battery discharge power
            if power_battery[i] >= discharge_power_battery:
                power_battery[i] = discharge_power_battery

            # If fuel cell power is below the minimum threshold, adjust the battery
            if power_demand[i] - power_battery[i] <= fuel_cell_min_power:
                power_fuel_cell[i] = fuel_cell_min_power
            else: 
                power_fuel_cell[i] = power_demand[i] - power_battery[i]

                
                # power_battery[i] = power_demand[i]

        elif power_demand[i] < 0:  # Battery charging
            charging_power = -power_demand[i]

            if SoC[i - 1] < 55:  # SoC < 55% --> 10C charging rate
                max_charging_power = charge_power_battery_10C
            elif SoC[i - 1] >= 55:  # SoC > 55% --> 6C charging rate
                max_charging_power = charge_power_battery_6C
            # Cap the charging power
            if charging_power > max_charging_power:
                charging_power = max_charging_power

            power_battery[i] = -charging_power  # Charging, negative power

        # Update SoC based on your formula

        SoC[i] = min(SoC_max, max(SoC_min, SoC[i - 1] - ((power_battery[i] * 1 / 3600) / battery_capacity) +
                        (( -power_demand[i] * 1 / 3600) / battery_capacity)))

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
def plot_results(t, v, v_s, acceleration, Fair, Frolling, F_total, InP, SoC, power_battery, power_fuel_cell, power_hybrid,Bat_motor_gen,Bat_motor_demand,InP_Hybrid):
    time = len(t)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(5, 1, 1)
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

    plt.subplot(5, 1, 2)
    plt.plot(t, F_total / 1000, 'k-', label='Total Force Resistance (kN)')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Force Resistance (kN)')
    plt.xlim([t[0], time])
    plt.annotate(f'Min: {min(F_total / 1000):.2f} kN', xy=(t[np.argmin(F_total)], min(F_total / 1000)), 
                 xytext=(t[np.argmin(F_total)], min(F_total / 1000) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(F_total / 1000):.2f} kN', xy=(t[np.argmax(F_total)], max(F_total / 1000)), 
                 xytext=(t[np.argmax(F_total)], max(F_total / 1000) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.legend()
    
    plt.subplot(5, 1, 3)  # 7 rows, 1 column, sixth subplot
    
    plt.plot(t, InP_Hybrid/1000, 'g-', label='Instant Power (kW)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Instant Power (kW)')
    plt.annotate(f'Min: {min(InP_Hybrid / 1000):.2f} kW', xy=(t[np.argmin(InP_Hybrid)], min(InP_Hybrid / 1000)), 
                 xytext=(t[np.argmin(InP_Hybrid)], min(InP_Hybrid / 1000) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(InP_Hybrid / 1000):.2f} kW', xy=(t[np.argmax(InP_Hybrid)], max(InP_Hybrid / 1000)), 
                 xytext=(t[np.argmax(InP_Hybrid)], max(InP_Hybrid / 1000) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    
    plt.legend()
    
    plt.subplot(5, 1, 4)  # 7 rows, 1 column, sixth subplot
    
    plt.plot(t, Bat_motor_demand/1000, 'b-', label=' provided power (kW) ')
    plt.xlim([0, time])
    # plt.ylim([min(Bat_motor_demand/1000), max(Bat_motor_demand/1000)])
    plt.xlabel('Time (s)')
    plt.ylabel('Power (kW)')
    plt.annotate(f'Min: {min(Bat_motor_demand / 1000):.2f} kW', xy=(t[np.argmin(Bat_motor_demand)], min(Bat_motor_demand / 1000)), 
                 xytext=(t[np.argmin(Bat_motor_demand)], min(Bat_motor_demand / 1000) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(Bat_motor_demand / 1000):.2f} kW', xy=(t[np.argmax(Bat_motor_demand)], max(Bat_motor_demand / 1000)), 
                 xytext=(t[np.argmax(Bat_motor_demand)], max(Bat_motor_demand / 1000) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    
    plt.legend()
    
    plt.subplot(5, 1, 5)  # 7 rows, 1 column, sixth subplot
    
    plt.plot(t, Bat_motor_gen/1000, 'r-', label='Hybrid received power (kW)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')    
    plt.annotate(f'Min: {min(Bat_motor_gen / 1000):.2f} kW', xy=(t[np.argmin(Bat_motor_gen)], min(Bat_motor_gen / 1000)), 
                 xytext=(t[np.argmin(Bat_motor_gen)], min(Bat_motor_gen / 1000) + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.annotate(f'Max: {max(Bat_motor_gen / 1000):.2f} kW', xy=(t[np.argmax(Bat_motor_gen)], max(Bat_motor_gen / 1000)), 
                 xytext=(t[np.argmax(Bat_motor_gen)], max(Bat_motor_gen / 1000) - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle=' -> '), fontsize=8, color='black')
    plt.ylabel('Power (kW)')
    
    plt.legend()
    
    
    plt.tight_layout()
    plt.savefig('Question_A.png', dpi=200)
    plt.show()

    # Plot hybrid power and SoC
    plt.figure(figsize=(10, 8))

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
    InP, InP_Hybrid, Bat_motor_gen, Bat_motor_demand,F_total = calculate_power(t, v_s, acceleration, Fair, Frolling, Fcl)
    SoC, power_battery, power_fuel_cell, power_hybrid = simulate_soc_and_power(t, InP_Hybrid)

    plot_results(t, v, v_s, acceleration, Fair, Frolling, F_total, InP, SoC, power_battery, power_fuel_cell, power_hybrid,Bat_motor_gen,Bat_motor_demand,InP_Hybrid)

# Execute the main function
if __name__ == "__main__":
    main()
