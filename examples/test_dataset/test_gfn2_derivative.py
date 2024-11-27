import os
import subprocess

# Step 1: Read/Change Parameters Files
def read_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = file.readlines()
    return parameters

def write_parameters(file_path, parameters):
    with open(file_path, 'w') as file:
        file.writelines(parameters)

# Step 2: Read Results for F/E
def read_results(file_path):
    with open(file_path, 'r') as file:
        results = file.readlines()
    return results

def parse_results(results):
    forces = []
    energies = []
    for line in results:
        if 'Force' in line:
            forces.append(float(line.split()[1]))
        elif 'Energy' in line:
            energies.append(float(line.split()[1]))
    return forces, energies

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
    param_file = '/scratch/pawsey0799/yx7184/xtb_install/share/xtb/param_gfn2-xtb.txt'
    result_file = 'param_gfn2-xtb.txt'
    fortran_executable = './fortran_program'
    input_file = 'input.dat'
    output_file = 'output.dat'

    derivatives = numerical_derivative_wrapper(param_file, result_file, fortran_executable, input_file, output_file)
    print(derivatives)
