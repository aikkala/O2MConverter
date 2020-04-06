import numpy as np
from pyquaternion import Quaternion
import math
import pandas as pd
import matplotlib.pyplot as pp
import mujoco_py
import skvideo.io


def is_nested_field(d, field, nested_fields):
    # Check if field is in the given dictionary after nested fields
    if len(nested_fields) > 0:
        if nested_fields[0] in d:
            return is_nested_field(d[nested_fields[0]], field, nested_fields[1:])
        else:
            return False
    else:
        if field in d:
            return True
        else:
            return False


def create_rotation_matrix(axis, rad=None, deg=None):
    R = np.eye(4, 4)

    # Make sure axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    l = axis[0]
    m = axis[1]
    n = axis[2]

    # Convert deg to rad if needed
    if rad is None:
        if deg is None:
            raise ValueError("Either rad or deg must be given")
        rad = (math.pi / 180) * deg

    # Create the rotation matrix
    R[0, 0] = l*l*(1-np.cos(rad)) + np.cos(rad)
    R[0, 1] = m*l*(1-np.cos(rad)) - n*np.sin(rad)
    R[0, 2] = n*l*(1-np.cos(rad)) + m*np.sin(rad)
    R[1, 0] = l*m*(1-np.cos(rad)) + n*np.sin(rad)
    R[1, 1] = m*m*(1-np.cos(rad)) + np.cos(rad)
    R[1, 2] = n*m*(1-np.cos(rad)) - l*np.sin(rad)
    R[2, 0] = l*n*(1-np.cos(rad)) - m*np.sin(rad)
    R[2, 1] = m*n*(1-np.cos(rad)) + l*np.sin(rad)
    R[2, 2] = n*n*(1-np.cos(rad)) + np.cos(rad)

    return R


def create_translation_vector(axis, l):
    t = np.zeros(shape=(3,))
    if axis[0] == 1 and axis[1] == 0 and axis[2] == 0:
        t[0] = l
    elif axis[0] == 0 and axis[1] == 1 and axis[2] == 0:
        t[1] = l
    elif axis[0] == 0 and axis[1] == 0 and axis[2] == 1:
        t[2] = l
    else:
        raise NotImplementedError
    return t


def create_translation_matrix(axis, l):
    T = np.eye(4, 4)
    t = create_translation_vector(axis, l)
    T[0:3, 3] = t
    return T


def create_symmetric_matrix(vec):
    # Assume vec is a vector of upper triangle values for matrix of size 3x3 (xx,yy,zz,xy,xz,yz)
    matrix = np.diag(vec[0:3])
    matrix[0, 1] = vec[3]
    matrix[0, 2] = vec[4]
    matrix[1, 2] = vec[5]
    return matrix + matrix.T - np.diag(matrix.diagonal())


def array_to_string(array):
    return ' '.join(['%8g' % num for num in array])


def create_transformation_matrix(pos=None, quat=None, R=None):
    T = np.eye(4)

    if pos is not None:
        T[:3, 3] = pos

    if quat is not None:
        T[:3, :3] = Quaternion(quat).rotation_matrix
    elif R is not None:
        T[:3, :3] = R

    return T


def get_control(model, control_file):

    # Import muscle control values (force or control?) of reference movement
    control_values, control_header = parse_sto_file(control_file)

    # Make sure actuators match
    column_names = list(control_values)
    for muscle_name in model._actuator_name2id:
        if muscle_name not in column_names:
            print("Activations for muscle {} were not found from control file {}".format(muscle_name, control_file))
            return None

    return control_values, control_header


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


def reindex_dataframe(df, timestep, last_timestamp=None):
    # Reindex / interpolate dataframe with new timestep

    # Make time start from zero
    index_name = df.index.name
    df.index = df.index.values - df.index.values[0]

    # Get new index
    if last_timestamp is None:
        last_timestamp = df.index.values[-1]
    new_index = np.arange(0, last_timestamp+timestep, timestep)

    # Create a new dataframe
    new_df = pd.DataFrame(index=new_index)
    new_df.index.name = index_name

    # Copy and interpolate values
    for colname, col in df.iteritems():
        new_df[colname] = np.interp(new_index, df.index, col)

    return new_df


def estimate_joint_error(reference, simulated, joint_names=None, timesteps=None, plot=False, output_file=None):

    # Reference and simulated need to be same shape
    if reference.shape != simulated.shape:
        print("Shapes of reference and simulated must be equal")
        return np.nan

    # Collect errors per joint
    errors = np.empty((reference.shape[1],))

    if plot:
        fig, axes = pp.subplots(reference.shape[1], 1, figsize=(10, 18))
        if timesteps is None:
            axes[-1].set_xlabel('Time (indices)')
            timesteps = np.arange(0, reference.shape[0])
        else:
            axes[-1].set_xlabel('Time (seconds)')

    # Get abs sum of difference between reference joint and simulated joint
    for idx in range(reference.shape[1]):
        e = reference[:, idx] - simulated[:, idx]
        #errors[idx] = np.sum(np.abs(e))
        errors[idx] = np.matmul(e, e)
        if plot:
            axes[idx].plot(timesteps, e)
            axes[idx].plot(np.array([timesteps[0], timesteps[-1]]), np.array([0, 0]), 'k--')
            if joint_names is not None:
                axes[idx].set_ylabel(joint_names[idx])

    # Save the file if output_file is defined
    if plot and output_file is not None:
        fig.savefig(output_file)
        pp.close(fig)

    return errors


def check_muscle_order(model, data):

    # Make sure muscles are in the same order
    for run_idx in range(len(data)):
        if data[run_idx]["muscle_names"] != list(model.actuator_names):
            print("Muscles are in incorrect order, fix this")
            raise NotImplementedError


def get_target_state_indices(model, env):

    # Map mujoco joints to target joints
    target_state_indices = np.empty(len(env.target_states,), dtype=int)
    for idx, target_state in enumerate(env.target_states):
        target_state_indices[idx] = model.joint_names.index(target_state)

    return target_state_indices


def get_initial_states(model, env):

    # If there are initial states, reorder them according to model joints
    initial_states = None

    if env.initial_states is not None:
        qpos = np.zeros(len(model.joint_names),)
        qvel = np.zeros(len(model.joint_names),)
        ctrl = np.zeros(len(model.actuator_names),)

        # Get qpos and qvel for joints
        if "joints" in env.initial_states:
            for state in env.initial_states["joints"]:
                if state in model.joint_names:
                    idx = model.joint_names.index(state)
                    if "qpos" in env.initial_states["joints"][state]:
                        qpos[idx] = env.initial_states["joints"][state]["qpos"]
                    if "qvel" in env.initial_states["joints"][state]:
                        qvel[idx] = env.initial_states["joints"][state]["qvel"]

        # Get activations for actuators
        if "actuators" in env.initial_states:
            for actuator in env.initial_states["actuators"]:
                idx = model.actuator_names.index(actuator)
                ctrl[idx] = env.initial_states["actuators"][actuator]

        initial_states = {"qpos": qpos, "qvel": qvel, "ctrl": ctrl}

    return initial_states


def initialise_simulation(sim, timestep, initial_states=None):

    # Set timestep
    sim.model.opt.timestep = timestep

    # Reset sim
    sim.reset()

    # Set initial states
    if initial_states is not None:
        if "qpos" in initial_states:
            sim.data.qpos[:] = initial_states["qpos"]
        if "qvel" in initial_states:
            sim.data.qvel[:] = initial_states["qvel"]
        if "ctrl" in initial_states:
            sim.data.ctrl[:] = initial_states["ctrl"]

        # We might need to call forward to make sure everything is set properly after setting qpos (not sure if required)
        sim.forward()


def run_simulation(sim, controls, viewer=None, output_video_file=None):

    qpos = np.empty((len(controls), len(sim.model.joint_names)))

    # For recording / viewing purposes
    imgs = []
    width = 1200
    height = 1200

    # We assume there's one set of controls for each timestep
    for t in range(len(controls)):

        # Set muscle activations
        sim.data.ctrl[:] = controls[t, :]

        # Forward the simulation
        sim.step()

        # Get joint positions
        qpos[t, :] = sim.data.qpos

        if viewer is not None:
            if output_video_file is None:
                viewer.render()
            else:
                viewer.render(width, height, sim.model._camera_name2id["for_testing"])
                imgs.append(np.flipud(viewer.read_pixels(width, height, depth=False)))

    if viewer is not None and output_video_file is not None:

        # Get writer
        writer = skvideo.io.FFmpegWriter(output_video_file,
                                         inputdict={"-s": "{}x{}".format(width, height),
                                                    "-r": str(1/sim.model.opt.timestep)})
        # Write the video
        for img in imgs:
            writer.writeFrame(img)

        # Close writer
        writer.close()

    return qpos
