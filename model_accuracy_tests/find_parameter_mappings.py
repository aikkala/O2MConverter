import mujoco_py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
import cma

import Utils


def parse_sto_file(sto_file):

    with open(sto_file) as file:

        # Go through header and parse it
        header_found = False
        header = dict()
        for row in file:

            # Get rid of newline
            row = row.rstrip()

            # Check if there's an assignment
            if "=" in row:
                assignment = row.split("=")
                header[assignment[0]] = assignment[1]

            # Else ignore the header row
            else:
                columns = row.split("\n")
                if len(columns) == 1 and columns[0] == "endheader":
                    header_found = True
                    break

        if not header_found:
            print("Couldn't parse the header!")
            return pd.DataFrame.empty

        # Create a pandas dataframe from the actual data
        values = pd.read_csv(file, sep="\t", skipinitialspace=True, dtype=float)

        # Set time as index
        values = pd.DataFrame.set_index(values, "time")

        # Return the dataframe
        return values, header


def run_simulation(sim, model, kinematics_values, control_values, visualise=False):

    # Set initial joint positions
    for joint_name in model.joint_names:

        # Get initial value of reference movement for this joint
        value = kinematics_values[joint_name].iloc[0]

        # Set the initial value
        sim.data.qpos[model._joint_name2id[joint_name]] = value

    # We might need to call forward to make sure everything is set properly after setting qpos (not sure if required)
    sim.forward()

    if visualise:
        viewer = mujoco_py.MjViewer(sim)
    timesteps = control_values.index.values
    qpos = np.empty((timesteps.shape[0], len(model.joint_names)))
    for t, timestep in zip(range(len(timesteps)), timesteps):

        # Set muscle activations
        for muscle_name in model._actuator_name2id:
            sim.data.ctrl[model._actuator_name2id[muscle_name]] = control_values[muscle_name].iloc[t]

        sim.step()
        qpos[t, :] = sim.data.qpos
        if visualise:
            viewer.render()

    return {"qpos": pd.DataFrame(qpos, index=control_values.index, columns=model.joint_names)}


def get_control(model):

    # Import muscle control values (force or control?) of reference movement
    control_file = "/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_controls.sto"
    control_values, control_header = parse_sto_file(control_file)

    # Make sure actuators match
    column_names = list(control_values)
    for muscle_name in model._actuator_name2id:
        if muscle_name not in column_names:
            print("Activations for muscle {} were not found from control file {}".format(muscle_name, control_file))
            return None

    return control_values, control_header


def get_kinematics(model):

    # Import joint positions during reference movement
    kinematics_file = "/home/aleksi/Workspace/O2MConverter/models/opensim/Leg6Dof9Musc/CMC/leg6dof9musc_Kinematics_q.sto"
    kinematics_values, kinematics_header = parse_sto_file(kinematics_file)

    # TODO Make sure joints match

    # Transform joint degrees into radians
    if kinematics_header.get("inDegrees", "no") == "yes":
        for joint_name in model._joint_name2id:
            if model.jnt_type[model._joint_name2id[joint_name]] == 3:
                kinematics_values[joint_name] = kinematics_values[joint_name] * (math.pi/180)

    return kinematics_values, kinematics_header


def update_equality_constraint(model, kinematics_values):
    # We need to
    # 1) find the equality constraint that is between worldbody and origin body
    # 2) find the joints that are between worldbody and origin body
    # 3) find the initial positions of reference movement for those joints
    # 4) update the location and orientation of the equality constraint with those positions

    ## 1)
    # Find id of world body
    world_id = model._body_name2id["world"]

    # Find equality constraint where world body is obj2
    eq_id = np.where(model.eq_obj2id == world_id)[0]

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
        value = kinematics_values[model._joint_id2name[joint_id]].iloc[0]

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

    # Get transformation from origin body (with joint values initialised) to world body
    T = np.linalg.solve(T, Tm)

    # Update location and orientation of the weld constraint
    q = Quaternion(matrix=T)
    model.eq_data[0][:3] = T[:3, 3]
    model.eq_data[0][3:] = np.array([q.w, q.x, q.y, q.z])

    # Set equality constraint on
    model.eq_active[eq_id] = 1

    # Relatively small (and non-zero) values seem to suppress movement about the weld constraint. Not sure why?
    model.dof_armature[:3] = 1
    model.jnt_stiffness[:3] = 1


def reindex_dataframe(df, timestep):
    # Reindex / interpolate dataframe with new timestep

    # Make time start from zero
    index_name = df.index.name
    df.index = df.index.values - df.index.values[0]

    # Get new index
    new_index = np.arange(0, df.index.values[-1], timestep)

    # Create a new dataframe
    new_df = pd.DataFrame(index=new_index)
    new_df.index.name = index_name

    # Copy and interpolate values
    for colname, col in df.iteritems():
        new_df[colname] = np.interp(new_index, df.index, col)

    return new_df


def estimate_joint_error(reference, simulated):
    # Get joint names
    joint_names = simulated.columns.tolist()

    # Get abs sum of difference between reference joint and simulated joint
    errors = np.empty((len(joint_names),))
    for joint_name, idx in zip(joint_names, range(len(joint_names))):
        errors[idx] = (reference[joint_name] - simulated[joint_name]).abs().sum()

    return errors


def get_model_params(model, param_names):
    pass


def main():

    # Define XML file
    model_xml_path = "/home/aleksi/Workspace/O2MConverter/models/converted/leg6dof9musc_converted/leg6dof9musc_converted.xml"

    # Initialise MuJoCo with the converted model
    model = mujoco_py.load_model_from_path(model_xml_path)

    # Get muscle control values
    control_values, control_header = get_control(model)
    if control_values is None:
        return

    # Get joint kinematics values
    kinematics_values, kinematics_header = get_kinematics(model)
    if kinematics_values is None:
        return

    # Make sure both muscle control and joint kinematics have the same timesteps
    if not control_values.index.equals(kinematics_values.index):
        print("Timesteps do not match between muscle control and joint kinematics")
        return

    # Timestep might not be constant in the OpenSim reference movement (weird). We can't change timestep dynamically in
    # mujoco, at least the viewer does weird things and it could be reflecting underlying issues. Thus, we should
    # to fit a spline and interpolate the muscle control and joint kinematics with model.opt.timestep
    control_values = reindex_dataframe(control_values, model.opt.timestep)
    kinematics_values = reindex_dataframe(kinematics_values, model.opt.timestep)

    # We need to set the location and orientation of the equality constraint to match the starting position of the
    # reference motion.
    update_equality_constraint(model, kinematics_values)

    # Initialise simulation
    sim = mujoco_py.MjSim(model)

    # Run CMA-ES while error is large or we haven't reached repetition limit
    N = 100
    E = 1e-6
    errors = np.empty((N, len(model.joint_names)))
#    for k in range(N):
#        sim.reset()
#        output = run_simulation(sim, model, kinematics_values, control_values)
#        errors[k, :] = estimate_joint_error(kinematics_values, output["qpos"])
#        if np.sum(errors[k, :]) < E:
#            break

    # Get initial values for params
    params = [1000]*9
    es = cma.CMAEvolutionStrategy(params, 20, {"transformation": [lambda x: abs(x), None]})

    while not es.stop():
        solutions = es.ask()

        fitness = []
        for solution in solutions:
            #model.dof_damping[3:] = solution[0]
            #model.dof_armature[3:] = solution[1]
            #model.jnt_stiffness[3:] = solution[2]
            for muscle_idx in range(model.actuator_gainprm.shape[0]):
                model.actuator_gainprm[muscle_idx][3] = solution[muscle_idx]

            sim.reset()

            output = run_simulation(sim, model, kinematics_values, control_values)
            f = np.sum(estimate_joint_error(kinematics_values, output["qpos"])) + 0.00001*np.sum(np.square(solution))
            fitness.append(f)

        es.tell(solutions, fitness)
        es.logger.add()
        es.disp()

    for muscle_idx in range(model.actuator_gainprm.shape[0]):
        model.actuator_gainprm[muscle_idx][3] = es.result.xbest[muscle_idx]
    sim.reset()
    run_simulation(sim, model, kinematics_values, control_values, visualise=True)

    print(es.result_pretty())
    es.plot()

main()
