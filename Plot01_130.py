import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading the Excel file with proper try-except handling
try:
    df = pd.read_excel('Plot01_130.xlsx', sheet_name='Sheet1')
    t = df.iloc[:, 0]  # Assuming 1st column is time or X-axis
    v = df.iloc[:, 1]  # Assuming 2nd column is speed in km/h or Y-axis
    alpha = 2 # 2 degree
    time = 300
    # Filter data to include only time between 0 and 1800
    mask = (t >= 0) & (t <= len(t))
    t = t[mask]
    v = v[mask]

    # Convert speed to m/s
    v_s = v * 1000 / 3600

    # Calculate the difference in velocity and time
    delta_v = np.diff(v_s)
    delta_t = np.diff(t)

    # Calculate acceleration (a = Δv / Δt)
    acceleration = delta_v / delta_t

    # Calculate air resistance (Fair = 0.5 * rho * v^2 * Cd * A)
    Fair = 0.5 * 1.2 * v_s**2 * 0.29 * 2.25
    Fair_df = pd.DataFrame({
        'Time': t,
        'Fair': Fair
    })

    # Calculate rolling resistance (Frolling = m * Cr * g * cos(theta))
    Frolling = 2000 * 0.0115 * 9.81 * np.cos(alpha)  # constant for all time points
    Frolling_df = pd.DataFrame({
        'Time': t,
        'Frolling': [Frolling] * len(t)
    })

    # Calculate climbing resistance (Fclimbing = m * g * sin(theta))
    Fcl = 2000 * 9.81 * np.sin(alpha)  # assuming no incline (theta = 0)
    Fcl_df = pd.DataFrame({
        'Time': t,
        'Fcl': [Fcl] * len(t)
    })

    # Calculate total force (F = Fcl + Frolling + Fair + ma)
    F_total = Fcl_df['Fcl'][1:] + Frolling_df['Frolling'][1:] + Fair_df['Fair'][1:] + 2000 * acceleration
    InP = v_s * F_total  # in Watts
    # InP[0] = 0
    InP[0] = np.int64(0)
    # DC/DC converter
    converter_efficiency = 0.9
    aux_output_power = 300  # in Watts

    Pgen = np.zeros(len(t))
    Pdemand = np.zeros(len(t))
    Bat_motor_gen = np.zeros(len(t))
    Bat_motor_demand = np.zeros(len(t))
    InP_Hybrid = np.zeros(len(t))
    Bat = aux_output_power / converter_efficiency

    for i in range(len(t)):
        if InP[i] <= 0:
            Pgen[i] = InP[i]
            Bat_motor_gen[i] = Pgen[i] * converter_efficiency
        elif InP[i] >= 0:
            Pdemand[i] = InP[i]
            Bat_motor_demand[i] = Pdemand[i] / converter_efficiency  # in Watts

    for i in range(len(t)):
        InP_Hybrid[i] = Bat + Bat_motor_gen[i] + Bat_motor_demand[i]  # in Watts
        
        
    plt.figure(figsize=(10, 6))
    # Plot total force (F_total)
    plt.subplot(3, 1, 1)  # 7 rows, 1 column, sixth subplot
    
    plt.plot(t, Bat_motor_gen/1000, 'r-', label='Hybrid received power (kW)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    # plt.ylim([min(Bat_motor_gen/1000), max(Bat_motor_gen/1000)])
    plt.ylabel('Hybrid power (kW)')
    plt.legend()
    
    plt.subplot(3, 1, 2)  # 7 rows, 1 column, sixth subplot
    
    plt.plot(t, Bat_motor_demand/1000, 'b-', label='Hybrid provided power (kW) ')
    plt.xlim([0, time])
    # plt.ylim([min(Bat_motor_demand/1000), max(Bat_motor_demand/1000)])
    plt.xlabel('Time (s)')
    plt.ylabel('Hybrid power (kW)')
    plt.legend()
    
    plt.subplot(3, 1, 3)  # 7 rows, 1 column, sixth subplot
    
    plt.plot(t, InP_Hybrid/1000, 'g-', label='Instant Power (kW)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Instant Power (kW)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Question_B_130.png', dpi=200)
    plt.show()
    
    # ============================================================================================
    
    # Battery li-ion store 1.24kWh
    battery_capacity = 1.24  # Battery capacity in Wh
    SoC_min = 50  # Minimum SoC percentage
    SoC_max = 65  # Maximum SoC percentage
    discharge_power_battery = 12.4   # Discharge power of the battery in Watts (10C)
    charge_power_battery_10C = 12.4   # Charge power when SoC < 55% (10C)
    charge_power_battery_6C = 7.44   # Charge power when SoC > 55% (6C)
    fuel_cell_min_power = 2.5   # Minimum fuel cell power in Watts

    # Initial conditions
    SoC = np.zeros(len(t))
    SoC[0] = 60  # Start with maximum SoC

    power_battery = np.zeros(len(t))
    power_fuel_cell = np.zeros(len(t))
    power_hybrid = np.zeros(len(t))
    power_fuel_cell[0] = fuel_cell_min_power
    # Power demand
    power_demand = InP/1000

    # Simulation loop for both charging and discharging
    for i in range(1, len(t)):
        if power_demand[i] > 0:  # Battery discharging
            if SoC[i - 1] < 55:  # SoC < 55% --> 10C discharge rate
                power_battery[i] = 0.05 * power_demand[i]  # 5% of total demand
                charging_power = charge_power_battery_10C
            elif SoC[i - 1] > 55:  # SoC > 55% --> 6C discharge rate
                power_battery[i] = 0.30 * power_demand[i]  # 30% of total demand
                charging_power = charge_power_battery_6C

            # Cap battery discharge power
            if power_battery[i] > discharge_power_battery:
                power_battery[i] = discharge_power_battery

            # Remaining power provided by the fuel cell

            power_fuel_cell[i] = power_demand[i] - power_battery[i]

            # If fuel cell power is below the minimum threshold, adjust the battery
            if power_fuel_cell[i] <= fuel_cell_min_power:

                power_fuel_cell[i] = fuel_cell_min_power

                # power_battery[i] = power_demand[i]

        elif power_demand[i] < 0:  # Battery charging
            charging_power = -power_demand[i]

            if SoC[i - 1] < 55:  # SoC < 55% --> 10C charging rate
                max_charging_power = charge_power_battery_10C
            elif SoC[i - 1] > 55:  # SoC > 55% --> 6C charging rate
                max_charging_power = charge_power_battery_6C

            # Cap the charging power
            if charging_power > max_charging_power:
                charging_power = max_charging_power

            power_battery[i] = -charging_power  # Charging, negative power

        # Update SoC based on your formula

        SoC[i] = min(SoC_max, max(SoC_min, SoC[i - 1] - ((power_battery[i] * 1 / 3600) / battery_capacity) +
                        (( -power_demand[i] * 1 / 3600) / battery_capacity)))

        # Ensure SoC remains within bounds
        if SoC[i] <= SoC_min:
                SoC[i] = SoC_min  # Cap at minimum SoC
                power_battery[i] = 0  # Prevent further discharge
        elif SoC[i] >= SoC_max:
                SoC[i] = SoC_max  # Cap at maximum SoC
                power_battery[i] = 0  # Prevent further charging
        power_hybrid[i] = power_battery[i] + power_fuel_cell[i]

    
    # Create the first figure with subplots
    plt.figure(figsize=(10, 15))

    # Plot speed in km/h
    plt.subplot(5, 1, 1)  
    plt.plot(t, v, 'r-', label='Speed (km/h)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    plt.text(t[v.argmax()], v.max(), f'Max: {v.max():.1f} km/h', 
            horizontalalignment='center', verticalalignment='bottom')
    plt.text(t[v.argmin()], v.min(), f'Min: {v.min():.1f} km/h', 
            horizontalalignment='center', verticalalignment='top')

    # Plot speed in m/s
    plt.subplot(5, 1, 2)  
    plt.plot(t, v_s, 'b-', label='Speed (m/s)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.text(t[v_s.argmax()], v_s.max(), f'Max: {v_s.max():.1f} m/s', 
            horizontalalignment='center', verticalalignment='bottom')
    plt.text(t[v_s.argmin()], v_s.min(), f'Min: {v_s.min():.1f} m/s', 
            horizontalalignment='center', verticalalignment='top')

    # Plot acceleration
    plt.subplot(5, 1, 3)  
    plt.plot(t[1:], acceleration, 'g-', label='Acceleration (m/s^2)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.text(t[1:][acceleration.argmax()], acceleration.max(), f'Max: {acceleration.max():.1f} m/s²', 
            horizontalalignment='center', verticalalignment='bottom')
    plt.text(t[1:][acceleration.argmin()], acceleration.min(), f'Min: {acceleration.min():.1f} m/s²', 
            horizontalalignment='center', verticalalignment='top')

    # Plot air resistance (Fair)
    plt.subplot(5, 1, 4)  
    plt.plot(t, Fair/1000, 'y-', label='Air Resistance (kN)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Air Resistance (kN)')
    plt.legend()
    plt.text(t[Fair.argmax()], Fair.max()/1000, f'Max: {Fair.max()/1000:.1f} kN', 
            horizontalalignment='center', verticalalignment='bottom')
    plt.text(t[Fair.argmin()], Fair.min()/1000, f'Min: {Fair.min()/1000:.1f} kN', 
            horizontalalignment='center', verticalalignment='top')

    # Plot rolling resistance (Frolling)
    plt.subplot(5, 1, 5)  
    plt.plot(t, Frolling_df['Frolling']/1000, 'k-', label='Rolling Resistance (kN)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Rolling Resistance (kN)')
    plt.legend()
    plt.text(t[Frolling_df['Frolling'].argmax()], Frolling_df['Frolling'].max()/1000, f'Max: {Frolling_df["Frolling"].max()/1000:.1f} kN', 
            horizontalalignment='center', verticalalignment='bottom')
    plt.text(t[Frolling_df['Frolling'].argmin()], Frolling_df['Frolling'].min()/1000, f'Min: {Frolling_df["Frolling"].min()/1000:.1f} kN', 
            horizontalalignment='center', verticalalignment='top')

    plt.tight_layout()
    plt.savefig('Complete_Plots_130.png', dpi=200)
    plt.show()

    # Create the second figure
    plt.figure(figsize=(10, 8))

    # Plot total force (F_total)
    plt.subplot(2, 1, 1)  
    plt.plot(t[1:], F_total/1000, 'm-', label='Total Force (kN)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Total Force (kN)')
    plt.legend()
    # plt.text(t[1:][F_total.argmax()], F_total.max()/1000, f'Max: {F_total.max()/1000:.1f} kN',
    #         horizontalalignment='center', verticalalignment='bottom')
    # plt.text(t[1:][F_total.argmin()], F_total.min()/1000, f'Min: {F_total.min()/1000:.1f} kN',
    #         horizontalalignment='center', verticalalignment='top')

    # Plot power
    plt.subplot(2, 1, 2)  
    plt.plot(t, InP/1000, 'c-', label='Instant Power (kW)')
    plt.xlim([0, time])
    plt.xlabel('Time (s)')
    plt.ylabel('Power (kW)')
    plt.legend()
    # plt.text(t[InP.argmax()], InP.max()/1000, f'Max: {InP.max()/1000:.1f} kW',
    #         horizontalalignment='center', verticalalignment='bottom')
    # plt.text(t[InP.argmin()], InP.min()/1000, f'Min: {InP.min()/1000:.1f} kW',
    #         horizontalalignment='center', verticalalignment='top')

    plt.tight_layout()
    plt.savefig('Complete_Plots_130.png', dpi=200)
    plt.show()

    # Create the third figure
    plt.figure(figsize=(10, 8))

    # Plot hybrid power
    plt.subplot(4, 1, 1)
    plt.plot(t, power_hybrid, label="Hybrid Power (kW)")
    plt.xlim([0, time])
    plt.ylabel("Power (kW)")
    plt.legend()
    # plt.text(t[power_hybrid.argmax()], power_hybrid.max(), f'Max: {power_hybrid.max():.1f} kW',
    #         horizontalalignment='center', verticalalignment='bottom')
    # plt.text(t[power_hybrid.argmin()], power_hybrid.min(), f'Min: {power_hybrid.min():.1f} kW',
    #         horizontalalignment='center', verticalalignment='top')

    # Plot battery power
    plt.subplot(4, 1, 2)
    plt.plot(t, power_battery, label="Battery Power (kW)", color="orange")
    plt.xlim([0, time])
    plt.ylabel("Power (kW)")
    plt.legend()
    # plt.text(t[power_battery.argmax()], power_battery.max(), f'Max: {power_battery.max():.1f} kW',
    #         horizontalalignment='center', verticalalignment='bottom')
    # plt.text(t[power_battery.argmin()], power_battery.min(), f'Min: {power_battery.min():.1f} kW',
    #         horizontalalignment='center', verticalalignment='top')

    # Plot fuel cell power
    plt.subplot(4, 1, 3)
    plt.plot(t, power_fuel_cell, label="Fuel Cell Power (kW)", color="green")
    plt.xlim([0, time])
    plt.ylabel("Power (kW)")
    plt.legend()
    # plt.text(t[power_fuel_cell.argmax()], power_fuel_cell.max(), f'Max: {power_fuel_cell.max():.1f} kW',
    #         horizontalalignment='center', verticalalignment='bottom')
    # plt.text(t[power_fuel_cell.argmin()], power_fuel_cell.min(), f'Min: {power_fuel_cell.min():.1f} kW',
    #         horizontalalignment='center', verticalalignment='top')

    # Plot SoC
    plt.subplot(4, 1, 4)
    plt.plot(t, SoC, label="SoC (%)", color="red")
    plt.xlim([0, time])
    plt.ylabel("SoC (%)")
    plt.xlabel("Time (s)")
    plt.legend()
    # plt.text(t[SoC.argmax()], SoC.max(), f'Max: {SoC.max():.1f}%',
    #         horizontalalignment='center', verticalalignment='bottom')
    # plt.text(t[SoC.argmin()], SoC.min(), f'Min: {SoC.min():.1f}%',
    #         horizontalalignment='center', verticalalignment='top')

    plt.tight_layout()
    plt.savefig('Question_C_130.png', dpi=200)
    plt.show()

except FileNotFoundError:
    print("The file 'Plot01.xlsx' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
