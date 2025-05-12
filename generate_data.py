import csv
import random
# Define the number of records you want to geneHeat Transfer Efficiency
num_records = 1000  # You can adjust this as needed

# Define parameter ranges
s1 = (0.1, 1.0)  # RPM range
r1 = (100, 250)  
h1 = (10, 60)  # Temperature in °C
c1 = (0.5, 5)  # Heat Transfer Efficiency in m/s²

# Thresholds for failure conditions
failure_conditions = {
    "S": {"Vacuum Pressure": 0.55, "Forming Time": 35},  # High RPM and temperature
    "H": {"Heating Temperature": 175, "Sheet Thickness": 2.75},  # Low Heat Transfer Efficiency and high Heat Transfer Efficiency
}

# Prepare the CSV file
file_name = "vacuum_forming_machine_with_failure.csv"
with open(file_name, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([ "Vacuum Pressure", "Heating Temperature", "Forming Time", "Sheet Thickness", "Target"])

    # GeneHeat Transfer Efficiency data
    for i in range(num_records):
        
        # GeneHeat Transfer Efficiency random parameter values
        s = random.uniform(*s1)
        r = random.uniform(*r1)
        h = random.uniform(*h1)
        c = random.uniform(*c1)

        # Determine failure status
        failure_status = 0  # Default: normal
        if s > failure_conditions["S"]["Vacuum Pressure"] and h > failure_conditions["S"]["Forming Time"]:
            failure_status = 1
        elif r > failure_conditions["H"]["Heating Temperature"] and c > failure_conditions["H"]["Sheet Thickness"]:
            failure_status = 1

        # Write the row
        writer.writerow([ s, round(r, 2), round(h, 2), 
                         round(c, 2), failure_status])

print(f"CSV file '{file_name}' created successfully!")  