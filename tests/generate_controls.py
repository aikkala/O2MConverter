import sys
import mujoco_py
import numpy as np
import math
import csv
import os


def generate_controls(t, nactuators):
    # Generating completely random controls probably isn't the best idea, maybe slow sine waves with some added noise?
    # Modulate amplitude (with max 1), frequency, and phase
    min_amplitude = 0.02
    max_amplitude = 1.0
    max_frequency = 1.0

    # Initialise array
    controls = np.zeros((len(t), nactuators))

    # Generate a sine wave for each actuator
    for i in range(nactuators):
        if np.random.rand() > 0.3:
            continue
        freq = np.random.rand() * max_frequency
        amp = min_amplitude + np.random.rand() * (max_amplitude - min_amplitude)
        phase = np.random.rand() * 2*math.pi
        #phase = 0
        controls[:, i] = min_amplitude + (amp-min_amplitude)/2 * (np.sin(2*math.pi*freq*t + phase)+1)
        #controls[:, i] = (np.random.rand() * max_amplitude) * np.abs(np.sin(2 * math.pi * freq * t + phase))
        #controls[:, i] = np.linspace(0, (np.random.rand() * max_amplitude), controls.shape[0])

    return controls


def write_output(filepath, t, controls, actuators):

    # Open file
    with open(filepath, 'w') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write header first
        writer.writerow(["Random-generated controls"])
        writer.writerow(["version=1"])
        writer.writerow(["nRows={}".format(controls.shape[0])])
        writer.writerow(["nColumns={}".format(len(actuators) + 1)])
        writer.writerow(["inDegrees=no"])
        writer.writerow(["endheader"])

        # Then write column names
        writer.writerow(["time"] + actuators)

        # And finally the randomly generated control values
        np.savetxt(file, np.concatenate((t.reshape([-1, 1]), controls), axis=1), delimiter='\t', fmt='%12.8f')


def main(model_xml_path, N, output_folder="."):
    """Generate a set of controls (for both OpenSim and MuJoCo) for given MuJoCo model"""

    duration = 1
    timestep = 0.002
    t = np.arange(0, duration, timestep)

    # First we need to read the MuJoCo model file and get actuator names
    model = mujoco_py.load_model_from_path(model_xml_path)
    actuators = list(model.actuator_names)

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Then we have to generate a set of random controls for each actuator and write them to a OpenSim control file
    for i in range(N):

        # Generate controls
        controls = generate_controls(t, len(actuators))

        # Write to a file
        write_output(os.path.join(output_folder, "generated_controls_{}.sto".format(i)), t, controls, actuators)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
