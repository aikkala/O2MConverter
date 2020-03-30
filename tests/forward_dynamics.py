import pandas as pd
import mujoco_py
import math
import sys
import os
import Utils


def main(model_xml_path, control_folder):

    # Load the model
    model = mujoco_py.load_model_from_path(model_xml_path)

    # Initialise simulation
    sim = mujoco_py.MjSim(model)

    # Initialise viewer
    viewer = mujoco_py.MjViewer(sim)

    # Loop through control files
    control_files = os.listdir(control_folder)
    control_files = ["subject02_controls.sto"]
    for control_file in control_files:

        # Set initial values
        #sim.data.qpos[-3:] = [0, 0, 0]

        # Load forces
        control_values, control_header = Utils.get_control(model, os.path.join(control_folder, control_file))
        control_values = Utils.reindex_dataframe(control_values, model.opt.timestep)

        # Loop through timesteps and visualise
        for t in range(len(control_values)):

            # Set controls
            for actuator in model.actuator_names:
                sim.data.ctrl[model._actuator_name2id[actuator]] = control_values[actuator].iloc[t]

            # Step and visualise
            sim.step()
            viewer.render()


if __name__ == "__main__":
    main('/home/aleksi/Workspace/O2MConverter/models/converted/FullBodyModel_Hamner2010_v2_0_converted/FullBodyModel_Hamner2010_v2_0_converted.xml',
         '/home/aleksi/Workspace/O2MConverter/models/opensim/Hamner/outputReference/ResultsCMC')
