import sys
import mujoco_py
import numpy as np
import math
import csv
import os
from datetime import datetime
from tests.envs import EnvFactory


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
        if np.random.rand() > (1/3):
            continue
        freq = np.random.rand() * max_frequency
        amp = min_amplitude + np.random.rand() * (max_amplitude - min_amplitude)
        phase = np.random.rand() * 2*math.pi
        #phase = 0
        controls[:, i] = min_amplitude + (amp-min_amplitude)/2 * (np.sin(2*math.pi*freq*t + phase)+1)
        #controls[:, i] = (np.random.rand() * max_amplitude) * np.abs(np.sin(2 * math.pi * freq * t + phase))
        #controls[:, i] = np.linspace(0, (np.random.rand() * max_amplitude), controls.shape[0])

    return controls


def write_output(output_folder, filename, t, controls, actuators):

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open file
    with open(os.path.join(output_folder, filename), 'w') as file:
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


def get_epochtime_ms():
    return round(datetime.utcnow().timestamp() * 1000)


def main(model_name, N):
    """Generate a set of controls (for both OpenSim and MuJoCo) for given MuJoCo model"""

    # Get env
    env = EnvFactory.get(model_name)
    N = int(N)

    if env.opensim_timestep is not None:
        timestep = env.opensim_timestep
    else:
        timestep = env.timestep

    duration = 1
    t = np.arange(0, duration, timestep)

    # First we need to read the MuJoCo model file and get actuator names
    model = mujoco_py.load_model_from_path(env.mujoco_model_file)
    actuators = list(model.actuator_names)

    # Then we have to generate a set of random controls for each actuator and write them to a OpenSim control file
    for i in range(N):

        # Generate controls
        controls = generate_controls(t, len(actuators))

        # Create a folder and write to a file
        sub_folder = os.path.join(env.forward_dynamics_folder, "run_{}".format(get_epochtime_ms()))
        write_output(sub_folder, "controls.sto", t, controls, actuators)


if __name__ == "__main__":
    main(*sys.argv[1:])
    #main("mobl_arms", 100)
    #main("gait2392", 100)