import mujoco_py
import os
import math
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd


def extract_control_values(control_file):

    with open(control_file) as file:

        # We assume control_file is a csv file
        #csv_reader = csv.reader(file, delimiter="\t")

        # Go through header
        header_found = False
        for row in file:
            columns = row.rstrip().split("\n")
            if len(columns) == 1 and columns[0] == "endheader":
                header_found = True
                break

        if not header_found:
            print("Couldn't parse the header!")
            return pd.DataFrame.empty

        # Create a pandas dataframe from the actual data
        control_values = pd.read_csv(file, sep="\t", skipinitialspace=True, dtype=float)

        # Might as well set time as index
        #control_values.set_index(keys="time", inplace=True)

        # Return the dataframe
        return control_values


def main():
    # Import muscle control values (force or control?)
    control_file = "/home/allu/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_controls.sto"
    control_values = extract_control_values(control_file)

    model_xml_path = "/home/allu/Workspace/O2MConverter/models/converted/leg6dof9musc_converted/leg6dof9musc_converted.xml"

    # Initialise MuJoCo with the converted model
    mj_path = mujoco_py.utils.discover_mujoco()
    model = mujoco_py.load_model_from_path(model_xml_path)

    # Make sure numbers/names of actuators match

    # Get timestep differences
    timestep_diff = control_values["time"].diff()
    model.opt.timestep = 0.01

    sim = mujoco_py.MjSim(model)

#    viewer = mujoco_py.MjViewer(sim)
    t = 0
#    imgs = []
    while True:
        sim.model.opt.timestep = timestep_diff.iloc[t+1]
        sim.data.ctrl[0] = control_values["bifemlh_r"].iloc[t]
        sim.data.ctrl[1] = control_values["bifemsh_r"].iloc[t]
        sim.data.ctrl[2] = control_values["glut_max2_r"].iloc[t]
        sim.data.ctrl[3] = control_values["psoas_r"].iloc[t]
        sim.data.ctrl[4] = control_values["rect_fem_r"].iloc[t]
        sim.data.ctrl[5] = control_values["vas_int_r"].iloc[t]
        sim.data.ctrl[6] = control_values["med_gas_r"].iloc[t]
        sim.data.ctrl[7] = control_values["soleus_r"].iloc[t]
        sim.data.ctrl[8] = control_values["tib_ant_r"].iloc[t]

        t += 1
        plt.imsave("img_" + str(t) + ".png", sim.render(600, 600))
        sim.step()
#        viewer.render()
        if t > 500:
            break

    # Run with the same timesteps as controls from OpenSim, input muscle controls, output joint angles

    # Run CMA-ES to optimise mappings


main()
