import numpy as np
from pyquaternion import Quaternion
import math
import pandas as pd


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
