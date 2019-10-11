import mujoco_py
import os
import math
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

import Utils


def parse_sto_file(control_file):

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
    # Import muscle control values (force or control?) of reference movement
    control_file = "/home/allu/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_controls.sto"
    control_values = parse_sto_file(control_file)

    # Import joint positions during reference movement
    kinematics_file = "/home/allu/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_Kinematics_q.sto"
    kinematics_values = parse_sto_file(kinematics_file)

    # Define XML file
    model_xml_path = "/home/allu/Workspace/O2MConverter/models/converted/leg6dof9musc_converted/leg6dof9musc_converted.xml"

    # Initialise MuJoCo with the converted model
    model = mujoco_py.load_model_from_path(model_xml_path)

    # Make sure numbers/names of actuators match
    column_names = list(control_values)
    for muscle_name in model._actuator_name2id:
        if muscle_name not in column_names:
            print("Activations for muscle {} were not found from control file {}".format(muscle_name, control_file))
            return

    # We need to set the location and orientation of the equality constraint to match the starting position of the
    # reference motion. And for that we need to
    # 1) find the equality constraint that is between worldbody and origin body
    # 2) find the joints that are between worldbody and origin body
    # 3) find the initial positions of reference movement for those joints
    # 4) update the location and orientation of the equality constraint with those positions

    ## 1)
    # find id of world body
    world_id = model._body_name2id["world"]

    # find equality constraint where world body is obj2
    eq_id = np.where(model.eq_obj2id==world_id)[0]

    # Exit if equality constraint wasn't found
    if len(eq_id) == 0:
        print("Couldn't find equality constraint between world body and origin body")

    ## 2)
    # Find the (origin) body that is involved in the constraint with world body
    org_id = model.eq_obj1id[eq_id]

    # Find joints (indices) between origin body and world body
    joint_ids = np.where(model.jnt_bodyid == org_id)[0]

    ## 3)
    # Go through joints and do their transformations
    T = np.eye(4, 4)
    for joint_id in joint_ids:

        # Get initial value for this joint
        value = kinematics_values[model._joint_id2name[joint_id]].iloc(0)

        # Create a transformation matrix to the initial joint value
        if model.jnt_type[joint_id] == 2:
            # Slide joint
            T_joint = Utils.create_translation_matrix(model.jnt_axis[joint_id, :], value)
        elif model.jnt_type[joint_id] == 3:
            # Hinge joint
            T_joint = Utils.create_rotation_matrix(model.jnt_axis[joint_id, :], value)
        else:
            print("This type of joint has not been defined")
            return

        T = np.matmul(T, T_joint)

    ## 4)
    # Get transformation matrix from origin body to world body (i.e. relpose of the weld equality constraint)
    # NOTE! We assume that the weld constraint's position is equal to the origin body's position in world body
    # TODO check the above
    Tm = np.eye(4, 4)
    Tm[:3, 3] = model.eq_data[0][:3]
    Tm[:3, :3] = Quaternion(model.eq_data[0][3:]).rotation_matrix



    T = np.matmul(np.linalg.inv(np.matmul(np.matmul(R1, T1), T2)), Tm)

    q = Quaternion(matrix=T)

    model.eq_data[0][:3] = T[:3, 3]
    model.eq_data[0][3:] = np.array([q.w, q.x, q.y, q.z])

    # Set equality constraint on; make sure there's only one constraint
    if len(model.eq_active) > 1:
        print("There's supposed to be only one equality constraint, don't know how to handle this!")
        return
    model.eq_active[0] = 1

    # Relatively small (and non-zero) values seem to suppress movement about the weld constraint. Not sure why?
    model.dof_armature[:3] = 1
    model.jnt_stiffness[:3] = 1

    # Get timestep differences
    timestep_diff = control_values["time"].diff()

    sim = mujoco_py.MjSim(model)



    # Set initial positions; TODO make sure we're setting values to correct joints
    sim.data.qpos[0] = 0.0200194
    sim.data.qpos[1] = 0.058
    sim.data.qpos[2] = 1.060
    sim.data.qpos[3] = -0.17539924
    sim.data.qpos[4] = -0.6469763
    sim.data.qpos[5] = 0.09735134

    # we should update the location of equality constraint with these initial coordinates

#    viewer = mujoco_py.MjViewer(sim)
    t = 0
#    imgs = []
    while True:
        sim.model.opt.timestep = timestep_diff.iloc[t+1]
        #sim.data.ctrl[0] = control_values["bifemlh_r"].iloc[t]
        #sim.data.ctrl[1] = control_values["bifemsh_r"].iloc[t]
        #sim.data.ctrl[2] = control_values["glut_max2_r"].iloc[t]
        #sim.data.ctrl[3] = control_values["psoas_r"].iloc[t]
        #sim.data.ctrl[4] = control_values["rect_fem_r"].iloc[t]
        #sim.data.ctrl[5] = control_values["vas_int_r"].iloc[t]
        #sim.data.ctrl[6] = control_values["med_gas_r"].iloc[t]
        #sim.data.ctrl[7] = control_values["soleus_r"].iloc[t]
        #sim.data.ctrl[8] = control_values["tib_ant_r"].iloc[t]

        t += 1
        if t % 10 == 0:
            plt.imsave("img_" + str(t) + ".png", sim.render(600, 600))
        sim.step()

        #        viewer.render()
        if t > 500:
            break

    # Run with the same timesteps as controls from OpenSim, input muscle controls, output joint angles

    # Run CMA-ES to optimise mappings


main()
