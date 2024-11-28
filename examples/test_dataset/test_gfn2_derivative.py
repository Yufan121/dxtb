import os
import subprocess
import re

# Step 1: Read/Change Parameters Files
class ParameterFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.parameters = self.read_parameters()

    def read_parameters(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        return lines

    def write_parameters(self):
        with open(self.file_path, 'w') as file:
            file.writelines(self.parameters)

    def modify_parameter(self, section, key, new_value):
        section_found = False
        for i, line in enumerate(self.parameters):
            if line.strip().startswith(section):
                section_found = True
                print("".join(key.split()))
            elif section_found and line.startswith('$'):
                break
            elif (section_found and line.strip().startswith(key)) or (section_found and line.strip().startswith("".join(key.split()))):
                parts = line.split()
                if len(parts) - 1 != len(new_value.split()):
                    raise ValueError(f"New value dimensions do not match existing dimensions for key '{key}'.")
                indent = line[:line.index(key)]  # Preserve the original indentation
                self.parameters[i] = f"{indent}{key} {new_value}\n"
                return
        
        if not section_found:
            raise KeyError(f"Section '{section}' does not exist.")
        raise KeyError(f"Key '{key}' does not exist in section '{section}'.")




# Step 2: Read Results for F/E
def read_results(file_path):
    with open(file_path, 'r') as file:
        results = file.readlines()
    return results

def parse_results(results):
    energies = []
    for line in results:
        if 'TOTAL ENERGY' in line:
            energy_value = line.split()[3]  # Extract the energy value
            energies.append(float(energy_value))
    return energies


# Step 3: Calculate Derivative by Calling Fortran Program and Compute
def call_fortran_program(executable, input_file, output_file):
    subprocess.run([executable, input_file, output_file])

def compute_derivative(forces, energies):
    # Example: simple finite difference method
    derivatives = []
    for i in range(1, len(forces)):
        derivative = (forces[i] - forces[i-1]) / (energies[i] - energies[i-1])
        derivatives.append(derivative)
    return derivatives


# Wrapper Function
def numerical_derivative_wrapper(param_file, result_file, fortran_executable, input_file, output_file):
    # Step 1: Read/Change Parameters Files
    parameters = read_parameters(param_file)
    # Modify parameters as needed
    write_parameters(param_file, parameters)
    
    # Step 2: Read Results for F/E
    call_fortran_program(fortran_executable, input_file, output_file)
    results = read_results(result_file)
    forces, energies = parse_results(results)
    
    # Step 3: Calculate Derivative
    derivatives = compute_derivative(forces, energies)
    
    return derivatives


if __name__ == "__main__":
    
    # Example Usage
    fortran_executable = './fortran_program'
    input_file = 'input.dat'
    output_file = 'output.dat'

    # derivatives = numerical_derivative_wrapper(param_file, result_file, fortran_executable, input_file, output_file)
    # print(derivatives)

    # 2 param file change
    param_file = '/scratch/pawsey0799/yx7184/xtb_install/share/xtb/param_gfn2-xtb.txt'
    pf = ParameterFile(param_file)

    # Read parameters
    print("Read Parameters:")

    # Modify parameters (example: change ks value)
    try:
        pf.modify_parameter('$globpar', 'ks', '1.00000')
    except (KeyError, ValueError) as e:
        print(e)

    try: 
        pf.modify_parameter('$Z= 1', 'GAM3=', '1.500000') 
    except (KeyError, ValueError) as e: 
        print(e)
        
        
    # Write parameters back to file
    pf.write_parameters()
    print("\nModified Parameters Written to File")


    # 3 read output
    # Example usage
    file_path = 'output.log'
    results = read_results(file_path)
    energies = parse_results(results)
    print("Energies:", energies)
