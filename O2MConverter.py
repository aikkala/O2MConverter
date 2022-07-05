import numpy as np
import vtk
import sys
import os
from scipy.interpolate import interp1d
import math
import copy
import xmltodict
import admesh
from pyquaternion import Quaternion
from shutil import copyfile
from natsort import natsorted, ns
from operator import itemgetter
from sklearn.metrics import r2_score
from collections import OrderedDict
from scipy.optimize import minimize

import Utils


class Converter:
    """A class to convert OpenSim XML model files to MuJoCo XML model files"""

    def __init__(self):

        # Define input XML and output folder
        self.input_xml = None
        self.output_folder = None

        # List of constraints
        self.constraints = None

        # Parse bodies, joints, muscles
        self.bodies = dict()
        self.joints = dict()
        self.muscles = []

        # We need to keep track of coordinates in joints' CoordinateSet, we might need to use them for setting up
        # equality constraints
        self.coordinates = dict()

        # These dictionaries (or list of dicts) are in MuJoCo style (when converted to XML)
        self.asset = dict()
        self.tendon = []
        self.actuator = {"general": [], "muscle": []}
        self.equality = {"joint": [], "weld": []}

        # Use mesh files if they are given
        self.geometry_folder = None
        self.output_geometry_folder = "Geometry/"
        self.vtk_reader = vtk.vtkXMLPolyDataReader()
        self.stl_writer = vtk.vtkSTLWriter()

        # Setup writer
        self.stl_writer.SetInputConnection(self.vtk_reader.GetOutputPort())
        self.stl_writer.SetFileTypeToBinary()

        # The root of the kinematic tree
        self.origin_body = None
        self.origin_joint = None

    def reset(self):
        self.constraints = None
        self.bodies = dict()
        self.joints = dict()
        self.muscles = []
        self.coordinates = dict()
        self.asset = dict()
        self.tendon = []
        self.actuator = {"general": [], "muscle": []}
        self.equality = {"joint": [], "weld": []}
        self.origin_body = None
        self.origin_joint = None

    def convert(self, input_xml, output_folder, geometry_folder=None, for_testing=False):
        """Convert given OpenSim XML model to MuJoCo XML model"""

        # Reset all variables
        self.reset()

        # Save input and output XML files in case we need them somewhere
        self.input_xml = input_xml

        # Set geometry folder
        self.geometry_folder = geometry_folder

        # Read input_xml and parse it
        with open(input_xml) as f:
            text = f.read()
        p = xmltodict.parse(text)

        # Set output folder
        model_name = os.path.split(input_xml)[1][:-5] + "_converted"
        self.output_folder = output_folder + "/" + model_name + "/"

        # Create the output folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Find and parse constraints
        if "ConstraintSet" in p["OpenSimDocument"]["Model"] and p["OpenSimDocument"]["Model"]["ConstraintSet"]["objects"] is not None:
            self.parse_constraints(p["OpenSimDocument"]["Model"]["ConstraintSet"]["objects"])

        # Find and parse bodies and joints
        if "BodySet" in p["OpenSimDocument"]["Model"]:
            self.parse_bodies_and_joints(p["OpenSimDocument"]["Model"]["BodySet"]["objects"])

        # Find and parse muscles, and CoordinateLimitForces
        if "ForceSet" in p["OpenSimDocument"]["Model"]:
            self.parse_muscles_and_tendons(p["OpenSimDocument"]["Model"]["ForceSet"]["objects"])
            if "CoordinateLimitForce" in p["OpenSimDocument"]["Model"]["ForceSet"]["objects"]:
                self.parse_coordinate_limit_forces(p["OpenSimDocument"]["Model"]["ForceSet"]["objects"]["CoordinateLimitForce"])

        # If we're building this model for testing we need to unclamp all joints
        if for_testing:
            self.unclamp_all_mujoco_joints()

        # Now we need to re-assemble all of the above in MuJoCo format
        # (or actually a dict version of the model so we can use xmltodict to save the model into a XML file)
        mujoco_model = self.build_mujoco_model(p["OpenSimDocument"]["Model"]["@name"])

        # Add a camera for testing
        mujoco_model["mujoco"]["worldbody"]["camera"] = {"@name": "for_testing", "@pos": "0 0 0", "@euler": "0 0 0"}
        if for_testing:
            mujoco_model["mujoco"]["option"]["@collision"] = "predefined"
            del mujoco_model["mujoco"]["worldbody"]["geom"]

        # Finally, save the MuJoCo model into XML file
        output_xml = self.output_folder + model_name + ".xml"
        with open(output_xml, 'w') as f:
            f.write(xmltodict.unparse(mujoco_model, pretty=True, indent="  "))

        # We might need to fix stl files (if converted from OpenSim Geometry vtk files)
        if self.geometry_folder is not None:
            self.fix_stl_files()

    def parse_constraints(self, p):

        # Go through all (possibly different kinds of) constraints
        for constraint_type in p:

            # Make sure we're dealing with a list
            if isinstance(p[constraint_type], dict):
                p[constraint_type] = [p[constraint_type]]

            # Go through all constraints
            for constraint in p[constraint_type]:

                if "SimmSpline" in constraint["coupled_coordinates_function"] or \
                        "NaturalCubicSpline" in constraint["coupled_coordinates_function"]:

                    if "SimmSpline" in constraint["coupled_coordinates_function"]:
                        spline_type = "SimmSpline"
                    else:
                        spline_type = "NaturalCubicSpline"

                    # Get x and y values that define the spline
                    x_values = constraint["coupled_coordinates_function"][spline_type]["x"]
                    y_values = constraint["coupled_coordinates_function"][spline_type]["y"]

                    # Convert into numpy arrays
                    x_values = np.array(x_values.split(), dtype=float)
                    y_values = np.array(y_values.split(), dtype=float)

                    assert len(x_values) > 1 and len(y_values) > 1, "Not enough points, can't fit a spline"

                    # Fit a linear / quadratic / cubic / quartic function
                    fit = np.polynomial.polynomial.Polynomial.fit(x_values, y_values, min(4, len(x_values)-1))

                    # A simple check to see if the fit is alright
                    y_fit = fit(x_values)
                    assert r2_score(y_values, y_fit) > 0.5, "A bad approximation of the SimmSpline"

                    # Get the polynomial function's weights
                    polycoef = np.zeros((5,))
                    polycoef[:fit.coef.shape[0]] = fit.convert().coef

                elif "LinearFunction" in constraint["coupled_coordinates_function"]:

                    # Get coefficients of the linear function
                    coefs = np.array(constraint["coupled_coordinates_function"]["LinearFunction"]["coefficients"].split(), dtype=float)

                    # Make a quartic representation of the linear function
                    polycoef = np.zeros((5,))
                    polycoef[0] = coefs[1]
                    polycoef[1] = coefs[0]

                    # Dummy linear fit function
                    fit = np.polynomial.polynomial.Polynomial.fit([0, 1], [0, 1], 1)

                else:
                    raise NotImplementedError

                # Create a constraint; note that 'fit' needs to be removed since it's not a valid mujoco keyword/string
                self.equality["joint"].append({
                    "@name": constraint["@name"],
                    "@joint1": constraint["dependent_coordinate_name"],
                    "@joint2": constraint["independent_coordinate_names"],
                    "@active": "true" if constraint["isDisabled"] == "false" else "false",
                    "@polycoef": Utils.array_to_string(polycoef),
                    "@solimp": "0.9999 0.9999 0.001 0.5 2",
                    "fit": fit})

    def parse_bodies_and_joints(self, p):

        # Go through all bodies and their joints
        for obj in p["Body"]:
            b = Body(obj)
            j = Joint(obj, self.equality)

            # Add b to bodies
            self.bodies[b.name] = b

            # Get coordinates, we might need them for setting up equality constraints
            self.coordinates = {**self.coordinates, **j.get_coordinates()}

            # Ignore joint if it is None
            if j.parent_body is None:
                continue

            # Add joint equality constraints
            self.equality["joint"].extend(j.get_equality_constraints("joint"))
            self.equality["weld"].extend(j.get_equality_constraints("weld"))

            # There might be multiple joints per body
            if j.parent_body not in self.joints:
                self.joints[j.parent_body] = []
            self.joints[j.parent_body].append(j)

    def parse_muscles_and_tendons(self, p):

        # Go through all muscle types (typically there are only one type of muscle)
        for muscle_type in p:

            # Skip some forces
            if muscle_type == "CoordinateLimitForce":
                # We'll handle these later
                continue
            elif muscle_type not in \
                    ["Millard2012EquilibriumMuscle", "Thelen2003Muscle",
                     "Schutte1993Muscle_Deprecated", "CoordinateActuator", "Millard2012AccelerationMuscle"]:
                print("Skipping a force: {}".format(muscle_type))
                continue

            # Make sure we're dealing with a list
            if isinstance(p[muscle_type], dict):
                p[muscle_type] = [p[muscle_type]]

            # Go through all muscles
            for muscle in p[muscle_type]:
                m = Muscle(muscle, muscle_type)
                self.muscles.append(m)

                # Check if the muscle is disabled
                if m.is_disabled():
                    continue
                elif m.is_muscle:
                    self.actuator["muscle"].append(m.get_actuator())
                    self.tendon.append(m.get_tendon())

                    # Add sites to all bodies this muscle/tendon spans
                    for body_name in m.path_point_set:
                        self.bodies[body_name].add_sites(m.path_point_set[body_name])
                else:
                    self.actuator["general"].append(m.get_actuator())

    def parse_coordinate_limit_forces(self, forces):

        # These parameters might be incorrect, but we'll optimize them later

        # Go through each force and set corresponding joint parameters
        for force in forces:

            # Ignore disabled forces
            if force["isDisabled"].lower() == "true":
                continue

            # Get joint name
            joint_name = force["coordinate"]

            # We need to search for this joint
            target = None
            for body in self.joints:
                for joint in self.joints[body]:
                    for mujoco_joint in joint.mujoco_joints:
                        if mujoco_joint["name"] == joint_name:
                            target = mujoco_joint

            # Check if joint was found
            assert target is not None, "Cannot set CoordinateLimitForce params, couldn't find the joint"

            # TODO for now let's ignore these forces -- they are too difficult to implement and optimize
            # Let's just switch the joint limit on if it's defined; mark this so it won't be unclamped later
            if "range" in target and target["range"][0] != target["range"][1]:
                target["limited"] = True
                target["user"] = 1
            continue

            # Take the average of stiffness
            stiffness = 0.5*(float(force["upper_stiffness"]) + float(force["lower_stiffness"]))

            # Stiffness / damping may be defined in two separate forces; we assume that we're dealing with damping
            # if average stiffness is close to zero
            if stiffness < 1e-4:

                # Check if rotational stiffness
                damping = float(force["damping"])
                if target["motion_type"] == "rotational":
                    damping *= math.pi/180

                # Set damping
                target["damping"] = damping

            else:

                # We need to create a soft joint coordinate limit, but we can't use separate limits like in OpenSim;
                # this is something we'll need to approximate

                # Limits in CoordinateLimitForce should be in degrees
                force_coordinate_limits = np.array([float(force["lower_limit"]), float(force["upper_limit"])]) * math.pi/180

                # Check if there are hard limits defined for this joint
                if target["limited"]:

                    # Range should be given if joint is limited; use range to calculate width param of solimp
                    range = target.get("range")
                    width = np.array([force_coordinate_limits[0] - range[0], range[1] - force_coordinate_limits[1]])

                    # If either width is > 0 create a soft limit
                    pos_idx = width > 0
                    if np.any(pos_idx):

                        # Mark this joint for optimization
                        target["user"] = 1

                        # Define the soft limit
                        target["solimplimit"] = [0.0001, 0.99, np.mean(width[pos_idx])]

                else:

                    # Use force_coordinate_limits as range

                    # Calculate width with the original range if it was defined
                    width = 0.001
                    if "range" in target:
                        width_range = np.array([force_coordinate_limits[0] - target["range"][0],
                                          target["range"][1] - force_coordinate_limits[1]])
                        pos_idx = width_range > 0
                        if np.any(pos_idx):
                            width = np.mean(width_range[pos_idx])

                    # Mark this joint for optimization
                    target["user"] = 1

                    # Define the soft limit
                    target["limited"] = True
                    target["solimplimit"] = [0.0001, 0.99, width, 0.5, 1]

    def build_mujoco_model(self, model_name):
        # Initialise model
        model = {"mujoco": {"@model": model_name}}

        # Set defaults
        # Note: balanceinertia is set to true, and boundmass and boundinertia are > 0 to ignore poorly designed models
        # (that contain incorrect inertial properties or massless moving bodies)
        model["mujoco"]["compiler"] = {"@inertiafromgeom": "auto", "@angle": "radian", "@balanceinertia": "true",
                                       "@boundmass": "0.001", "@boundinertia": "0.001"}
        model["mujoco"]["compiler"]["lengthrange"] = {"@inttotal": "50", "@useexisting": "true"}
        model["mujoco"]["default"] = {
            "joint": {"@limited": "true", "@damping": "0.5", "@armature": "0.1", "@stiffness": "2"},
            "geom": {"@rgba": "0.8 0.6 .4 1", "@margin": "0.001"},
            "site": {"@size": "0.001"},
            "tendon": {"@width": "0.001", "@rgba": ".95 .3 .3 1", "@limited": "false"}}
        model["mujoco"]["default"]["default"] = [
            {"@class": "muscle", "muscle": {"@ctrllimited": "true", "@ctrlrange": "0 1", "@scale": "200"}},
            {"@class": "motor", "general": {"@gainprm": "5 0 0 0 0 0 0 0 0 0"}}
            ]
        model["mujoco"]["option"] = {"@timestep": "0.002"}
        model["mujoco"]["size"] = {"@njmax": "1000", "@nconmax": "400", "@nuser_jnt": 1}

        # Start building the worldbody
        worldbody = {"geom": {"@name": "floor", "@pos": "0 0 0", "@size": "10 10 0.125",
                              "@type": "plane", "@material": "MatPlane", "@condim": "3"}}

        # We should probably find the "origin" body, where the kinematic chain begins
        self.origin_body, self.origin_joint = self.find_origin()

        # Rotate self.origin_joint.orientation_in_parent so the model is upright
        # Rotation is done along an axis that goes through (0,0,0) coordinate
        T_origin_joint = Utils.create_transformation_matrix(
            self.origin_joint.location_in_parent,
            quat=self.origin_joint.orientation_in_parent)
        T_rotation = Utils.create_rotation_matrix(axis=[1, 0, 0], rad=math.pi/2)
        self.origin_joint.set_transformation_matrix(np.matmul(T_rotation, T_origin_joint))

        # Add sites to worldbody / "ground" in OpenSim
        worldbody["site"] = self.bodies[self.origin_joint.parent_body].sites

        # Add some more defaults
        worldbody["body"] = {"light":
                                 {"@mode": "trackcom", "@directional": "false", "@pos": "0 0 4.0", "@dir": "0 0 -1"}}

        # Build the kinematic chains
        worldbody["body"] = self.add_body(worldbody["body"], self.origin_body,
                                          self.joints[self.origin_body.name])

        # Add worldbody to the model
        model["mujoco"]["worldbody"] = worldbody

        # We might want to use a weld constraint to fix the origin body to worldbody for experiments
        self.equality["weld"].append({"@name": "origin_to_worldbody",
                                      "@body1": self.origin_body.name, "@active": "false"})

        # Set some asset defaults
        self.asset["texture"] = [
            {"@name": "texplane", "@type": "2d", "@builtin": "checker", "@rgb1": ".2 .3 .4",
             "@rgb2": ".1 0.15 0.2", "@width": "100", "@height": "100"}]
        self.asset["material"] = [
            {"@name": "MatPlane", "@reflectance": "0.0", "@texture": "texplane",
             "@texrepeat": "1 1", "@texuniform": "true"}]

        # Add assets to model
        model["mujoco"]["asset"] = self.asset

        # Add tendons and actuators
        model["mujoco"]["tendon"] = {"spatial": self.tendon}
        model["mujoco"]["actuator"] = self.actuator

        # Add equality constraints between joints; note that we may need to remove some equality constraints
        # that were set in ConstraintSet but then overwritten or not used. While we're iterating through constraints
        # we also need to remove the "fit" keyword which is not a mujoco keyword/string
        remove_idxs = []
        for idx, constraint in enumerate(self.equality["joint"]):
            constraint_found = False
            if "fit" in constraint:
                del constraint["fit"]
            for parent_body in self.joints:
                for joint in self.joints[parent_body]:
                    for mujoco_joint in joint.mujoco_joints:
                        if mujoco_joint["name"] == constraint["@joint1"]:
                            constraint_found = True

            if not constraint_found:
                remove_idxs.append(idx)
                #self.equality["joint"].remove(constraint)

        # Remove constraints that aren't used
        for idx in sorted(remove_idxs, reverse=True):
            del self.equality["joint"][idx]

        # Add equality constraints into the model
        model["mujoco"]["equality"] = self.equality

        return model

    def unclamp_all_mujoco_joints(self):

        # Unclamp (set limited=false) all joints except those that have limites that need to be optimized

        for joint_name in self.joints:
            for j in self.joints[joint_name]:
                for mujoco_joint in j.mujoco_joints:
                    # Don't unclamp dependent joints or joints that have limits that need to be optimized
                    if mujoco_joint["motion_type"] not in ["dependent", "coupled"] \
                            and not ("user" in mujoco_joint and mujoco_joint["user"] == 1):
                        mujoco_joint["limited"] = False

    def add_body(self, worldbody, current_body, current_joints):

        # Create a new MuJoCo body
        worldbody["@name"] = current_body.name

        # We need to find this body's position relative to parent body:
        # since we're progressing down the kinematic chain, each body
        # should have a joint to parent body
        joint_to_parent = self.find_joint_to_parent(current_body.name)

        # Update location and orientation of child body
        T = Utils.create_transformation_matrix(joint_to_parent.location, quat=joint_to_parent.orientation)
        joint_to_parent.set_transformation_matrix(
            np.matmul(joint_to_parent.get_transformation_matrix(), np.linalg.inv(T)))

        # Define position and orientation
        worldbody["@pos"] = Utils.array_to_string(joint_to_parent.location_in_parent)
        worldbody["@quat"] = "{} {} {} {}"\
            .format(joint_to_parent.orientation_in_parent.w,
                    joint_to_parent.orientation_in_parent.x,
                    joint_to_parent.orientation_in_parent.y,
                    joint_to_parent.orientation_in_parent.z)

        # Add geom
        worldbody["geom"] = self.add_geom(current_body)

        # Add inertial properties -- only if mass is greater than zero and eigenvalues are positive
        # (if "inertial" is missing MuJoCo will infer the inertial properties from geom)
        if current_body.mass > 0:
            values, vectors = np.linalg.eig(Utils.create_symmetric_matrix(current_body.inertia))
            if np.all(values > 0):
                worldbody["inertial"] = {"@pos": Utils.array_to_string(current_body.mass_center),
                                         "@mass": str(current_body.mass),
                                         "@fullinertia": Utils.array_to_string(current_body.inertia)}

        # Add sites
        worldbody["site"] = current_body.sites

        # Go through joints
        worldbody["joint"] = []
        for mujoco_joint in joint_to_parent.mujoco_joints:

            # Define the joint
            j = {"@name": mujoco_joint["name"], "@type": mujoco_joint["type"], "@pos": "0 0 0",
                 "@axis": Utils.array_to_string(mujoco_joint["axis"])}
            if "limited" in mujoco_joint:
                j["@limited"] = "true" if mujoco_joint["limited"] else "false"
            if "range" in mujoco_joint:
                j["@range"] = Utils.array_to_string(mujoco_joint["range"])
            if "ref" in mujoco_joint:
                j["@ref"] = str(mujoco_joint["ref"])
            if "springref" in mujoco_joint:
                j["@springref"] = str(mujoco_joint["springref"])
            if "stiffness" in mujoco_joint:
                j["@stiffness"] = str(mujoco_joint["stiffness"])
            if "damping" in mujoco_joint:
                j["@damping"] = str(mujoco_joint["damping"])
            if "solimplimit" in mujoco_joint:
                j["@solimplimit"] = Utils.array_to_string(mujoco_joint["solimplimit"])
            if "user" in mujoco_joint:
                j["@user"] = str(mujoco_joint["user"])

            # If the joint is between origin body and it's parent, which should be "ground", set
            # damping, armature, and stiffness to zero
            if joint_to_parent is self.origin_joint:
                j.update({"@armature": 0, "@damping": 0, "@stiffness": 0})

            # Add to joints
            worldbody["joint"].append(j)

        # And we're done if there are no joints
        if current_joints is None:
            return worldbody

        worldbody["body"] = []
        for j in current_joints:
            worldbody["body"].append(self.add_body(
                {}, self.bodies[j.child_body],
                self.joints.get(j.child_body, None)
            ))

        return worldbody

    def add_geom(self, body):

        # Collect all geoms here
        geom = []

        if self.geometry_folder is None:

            # By default use a capsule
            # Try to figure out capsule size by mass or something
            size = np.array([0.01, 0.01])*np.sqrt(body.mass)
            geom.append({"@name": body.name, "@type": "capsule",
                         "@size": Utils.array_to_string(size)})

        else:

            # Make sure output geometry folder exists
            os.makedirs(self.output_folder + self.output_geometry_folder, exist_ok=True)

            # Grab the mesh from given geometry folder
            for m in body.mesh:

                # Get file path
                geom_file = self.geometry_folder + "/" + m["geometry_file"]

                # Check the file exists
                assert os.path.exists(geom_file) and os.path.isfile(geom_file), "Mesh file {} doesn't exist".format(geom_file)

                # Transform vtk into stl or just copy stl file
                mesh_name = m["geometry_file"][:-4]
                stl_file = self.output_geometry_folder + mesh_name + ".stl"

                # Transform a vtk file into an stl file and save it
                if geom_file[-3:] == "vtp":
                    self.vtk_reader.SetFileName(geom_file)
                    self.stl_writer.SetFileName(self.output_folder + stl_file)
                    self.stl_writer.Write()

                # Just copy stl file
                elif geom_file[-3:] == "stl":
                    copyfile(geom_file, self.output_folder + stl_file)

                else:
                    raise NotImplementedError("Geom file is not vtk or stl!")

                # Add mesh to asset
                self.add_mesh_to_asset(mesh_name, stl_file, m)

                # Create the geom
                geom.append({"@name": mesh_name, "@type": "mesh", "@mesh": mesh_name})

        return geom

    def add_mesh_to_asset(self, mesh_name, mesh_file, mesh):
        if "mesh" not in self.asset:
            self.asset["mesh"] = []
        self.asset["mesh"].append({"@name": mesh_name,
                                   "@file": mesh_file,
                                   "@scale": mesh["scale_factors"]})

    def find_origin(self):
        # Start from a random joint and work your way backwards until you find
        # the origin body (the body that represents ground)

        # Make sure there's at least one joint
        assert len(self.joints) > 0, "There are no joints!"

        # Choose a joint, doesn't matter which one
        current_joint = next(iter(self.joints.values()))[0]

        # Follow the kinematic chain
        while True:

            # Move up in the kinematic chain as far as possible
            new_joint_found = False
            for parent_body in self.joints:
                for j in self.joints[parent_body]:
                    if j.child_body == current_joint.parent_body:
                        current_joint = j
                        new_joint_found = True
                        break

            # No further joints, child of current joint is the origin body
            if not new_joint_found:
                return self.bodies[current_joint.child_body], current_joint

    def find_joint_to_parent(self, body_name):
        joint_to_parent = None
        for parent_body in self.joints:
            for j in self.joints[parent_body]:
                if j.child_body == body_name:
                    joint_to_parent = j

            # If there are multiple child bodies with the same name, the last
            # one is returned
            if joint_to_parent is not None:
                break

        assert joint_to_parent is not None, "Couldn't find joint to parent body for body {}".format(body_name)

        return joint_to_parent

    def fix_stl_files(self):
        # Loop through geometry folder and fix stl files
        for mesh_file in os.listdir(self.output_folder + self.output_geometry_folder):
            if mesh_file.endswith(".stl"):
                mesh_file = self.output_folder + self.output_geometry_folder + mesh_file
                stl = admesh.Stl(mesh_file)
                stl.remove_unconnected_facets()
                stl.write_binary(mesh_file)


class Joint:

    def __init__(self, obj, constraints):

        joint = obj["Joint"]
        self.parent_body = None
        self.coordinates = dict()

        # 'ground' body does not have joints
        if joint is None or len(joint) == 0:
            return

        # This code assumes there's max one joint per object
        assert len(joint) == 1, 'TODO Multiple joints for one body'

        # We need to figure out what kind of joint this is
        self.joint_type = list(joint)[0]

        # Step into the actual joint information
        joint = joint[self.joint_type]

        # Get names of bodies this joint connects
        self.parent_body = joint["parent_body"]
        self.child_body = obj["@name"]

        # And other parameters
        self.location_in_parent = np.array(joint["location_in_parent"].split(), dtype=float)
        self.location = np.array(joint["location"].split(), dtype=float)
        orientation = np.array(joint["orientation"].split(), dtype=float)
        x = Quaternion(axis=[1, 0, 0], radians=orientation[0]).rotation_matrix
        y = Quaternion(axis=[0, 1, 0], radians=orientation[1]).rotation_matrix
        z = Quaternion(axis=[0, 0, 1], radians=orientation[2]).rotation_matrix
        self.orientation = Quaternion(matrix=np.matmul(np.matmul(x, y), z))

        # Calculate orientation in parent
        orientation_in_parent = np.array(joint["orientation_in_parent"].split(), dtype=float)
        x = Quaternion(axis=[1, 0, 0], radians=orientation_in_parent[0]).rotation_matrix
        y = Quaternion(axis=[0, 1, 0], radians=orientation_in_parent[1]).rotation_matrix
        z = Quaternion(axis=[0, 0, 1], radians=orientation_in_parent[2]).rotation_matrix
        self.orientation_in_parent = Quaternion(matrix=np.matmul(np.matmul(x, y), z))

        # Not sure if we should update child body location and orientation before or after parsing joints;
        # at the moment we're doing it after
        #T = Utils.create_transformation_matrix(self.location, quat=self.orientation)
        #self.set_transformation_matrix(
        #    np.matmul(self.get_transformation_matrix(), np.linalg.inv(T)))

        # Some joint values are dependent on other joint values; we need to create equality constraints between those
        # Also we might need to use weld constraints on locked joints
        self.equality_constraints = {"joint": [], "weld": []}

        # CustomJoint can represent any joint, we need to figure out
        # what kind of joint we're dealing with
        self.mujoco_joints = []
        if self.joint_type == "CustomJoint":
            T_joint = self.parse_custom_joint(joint, constraints)

            # Update joint location and orientation
            T = self.get_transformation_matrix()
            T = np.matmul(T, T_joint)
            self.set_transformation_matrix(T)

        elif self.joint_type == "WeldJoint":
            # Don't add anything to self.mujoco_joints, bodies are by default
            # attached rigidly to each other in MuJoCo
            pass

        elif self.joint_type == "PinJoint":
            self.parse_pin_joint(joint)

        elif self.joint_type == "UniversalJoint":
            self.parse_universal_joint(joint)

        else:
            raise NotImplementedError

    def get_transformation_matrix(self):
        T = self.orientation_in_parent.transformation_matrix
        T[:3, 3] = self.location_in_parent
        return T

    def set_transformation_matrix(self, T):
        self.orientation_in_parent = Quaternion(matrix=T)
        self.location_in_parent = T[:3, 3]

    def parse_custom_joint(self, joint, constraints):
        # A CustomJoint in OpenSim model can represent any type of joint.
        # Try to parse the CustomJoint into a set of MuJoCo joints

        # Get transform axes
        transform_axes = joint["SpatialTransform"]["TransformAxis"]

        # We might need to create a homogeneous transformation matrix from
        # location_in_parent to actual joint location
        T = np.eye(4, 4)
        #T = self.orientation_in_parent.transformation_matrix

        # Start by parsing the CoordinateSet
        coordinate_set = self.parse_coordinate_set(joint)

        # NOTE! Coordinates in CoordinateSet parameterize this joint. In theory all six DoFs could be dependent
        # on one Coordinate. Here we assume that only one DoF is equivalent to a Coordinate, that is, there exists an
        # identity mapping between a Coordinate and a DoF, which is different to OpenSim where there might be no
        # identity mappings. In OpenSim a Coordinate is just a value and all DoFs might have some kind of mapping with
        # it, see e.g. "flexion" Coordinate in MoBL_ARMS_module6_7_CMC.osim model. MuJoCo doesn't have such abstract
        # notion of a "Coordinate", and thus there cannot be a non-identity mapping from a joint to itself

        # Go through axes; there's something wrong with the order of transformations, this is the order
        # that works for leg6dof9musc.osim and MoBL_ARMS_module6_7_CMC.osim models, but it's so weird
        # it's likely to be incorrect
        transforms = ["rotation1", "rotation2", "rotation3", "translation1", "translation2", "translation3"]
        order = [5, 4, 3, 0, 1, 2]
        #order = [0, 1, 2, 3, 4, 5]
        dof_designated = []
        for idx in order:

            t = transform_axes[idx]
            if t["@name"] != transforms[idx]:
                raise IndexError("Joints are parsed in incorrect order")

            # Use the Coordinate parameters we parsed earlier; note that these do not exist for all joints (e.g
            # constant joints)
            if t.get("coordinates", None) in coordinate_set:
                params = copy.deepcopy(coordinate_set[t["coordinates"]])
            else:
                params = {"name": "{}_{}".format(joint["@name"], t["@name"]), "limited": False,
                          "transform_value": 0, "coordinates": "unspecified"}

            params["original_name"] = params["name"]

            # Set default reference position/angle to zero. If this value is not zero, then you need
            # more care while calculating quartic functions for equality constraints
            params["ref"] = 0

            # By default add this joint to MuJoCo model
            params["add_to_mujoco_joints"] = True

            # See the comment before this loop. We have to designate one DoF per Coordinate as an independent variable,
            # i.e. make its dependence linear
            if "coordinates" in t and t["coordinates"] == params["name"] \
                    and t["@name"].startswith(params["motion_type"][:8]) and not params["name"] in dof_designated:

                # This is not necessary if the coordinate is dependent on another coordinate... starting to get
                # complicated
                ignore = False
                if "joint" in constraints:
                    for c in constraints["joint"]:
                        if params["name"] == c["@joint1"]:
                            ignore = True
                            break

                if not ignore:

                    # Check if we need to modify limits, TODO not sure if this is correct or needed
                    if Utils.is_nested_field(t, "SimmSpline", ["function"]):

                        # Fit a line/spline and check limit values within that fit
                        x_values = np.array(t["function"]["SimmSpline"]["x"].split(), dtype=float)
                        y_values = np.array(t["function"]["SimmSpline"]["y"].split(), dtype=float)
                        assert len(x_values) > 1 and len(y_values) > 1, "Not enough points, can't fit a polynomial"
                        fit = np.polynomial.polynomial.Polynomial.fit(x_values, y_values, min(4, len(x_values) - 1))
                        y_fit = fit(x_values)
                        assert r2_score(y_values, y_fit) > 0.5, "A bad approximation of the SimmSpline"

                        # Update range as min/max of the approximated range
                        if "range" in params:
                            params["range"] = fit(params["range"])
                        else:
                            params["range"] = np.array([min(y_fit), max(y_fit)])

                        # Make this into an identity mapping
                        t["function"] = dict({"LinearFunction": {"coefficients": '1 0'}})

                    elif Utils.is_nested_field(t, "LinearFunction", ["function"]):
                        coefficients = np.array(t["function"]["LinearFunction"]["coefficients"].split(), dtype=float)
                        assert abs(coefficients[0]) == 1 and coefficients[1] == 0, "Should we modify limits?"
                        x_values = np.array([0, 1])
                        y_values = np.array([0, 1])
                        fit = np.polynomial.polynomial.Polynomial.fit(x_values, y_values, 1)
                        y_fit = fit(x_values)

                    else:
                        raise NotImplementedError

                    # Joint values may have been mapped to a different range already, need to update constraints
                    if "joint" in constraints:
                        for c in constraints["joint"]:
                            if "@joint2" in c and params["name"] == c["@joint2"]:
                                assert c["fit"].degree() == fit.degree(), "Degrees must match"
                                fit = np.polynomial.polynomial.Polynomial.fit(y_fit, c["fit"](x_values), deg=c["fit"].degree())
                                polycoef = np.zeros((5,))
                                polycoef[:fit.coef.shape[0]] = fit.convert().coef
                                c["@polycoef"] = Utils.array_to_string(polycoef)

                    # Mark this dof as designated
                    dof_designated.append(params["name"])

            elif params["name"] in dof_designated:
                # A DoF has already been designated for a coordinate with params["name"], rename this joint
                params["name"] = "{}_{}".format(params["name"], t["@name"])

            # Handle a "Constant" transformation. We're not gonna create this joint
            # but we need the transformation information to properly align the joint
            flip_axis = False
            if Utils.is_nested_field(t, "Constant", ["function"]) or \
                    Utils.is_nested_field(t, "Constant", ["function", "MultiplierFunction", "function"]):

                # Get the value
                if "MultiplierFunction" in t["function"]:
                    value = float(t["function"]["MultiplierFunction"]["function"]["Constant"]["value"])
                elif "Constant" in t["function"]:
                    value = float(t["function"]["Constant"]["value"])
                else:
                    raise NotImplementedError

                # If the value is near zero don't bother creating this joint
                if abs(value) < 1e-6:
                    continue

                # Otherwise define a limited MuJoCo joint (we're not really creating this (sub)joint, we just update the
                # joint position)
                params["limited"] = True
                params["range"] = np.array([value])
                params["transform_value"] = value
                params["add_to_mujoco_joints"] = False

            # Handle a "SimmSpline" or "NaturalCubicSpline" transformation with a quartic approximation
            elif Utils.is_nested_field(t, "SimmSpline", ["function", "MultiplierFunction", "function"]) or \
                    Utils.is_nested_field(t, "NaturalCubicSpline", ["function", "MultiplierFunction", "function"]) or \
                    Utils.is_nested_field(t, "SimmSpline", ["function"]) or \
                    Utils.is_nested_field(t, "NaturalCubicSpline", ["function"]):

                # We can't model the relationship between two joints using a spline, but we can try to approximate it
                # with a quartic function. So fit a quartic function and check that the error is small enough

                # Get spline values
                if Utils.is_nested_field(t, "SimmSpline", ["function"]):
                    x_values = t["function"]["SimmSpline"]["x"]
                    y_values = t["function"]["SimmSpline"]["y"]
                elif Utils.is_nested_field(t, "NaturalCubicSpline", ["function"]):
                    x_values = t["function"]["NaturalCubicSpline"]["x"]
                    y_values = t["function"]["NaturalCubicSpline"]["y"]
                elif Utils.is_nested_field(t, "SimmSpline", ["function", "MultiplierFunction", "function"]):
                    x_values = t["function"]["MultiplierFunction"]["function"]["SimmSpline"]["x"]
                    y_values = t["function"]["MultiplierFunction"]["function"]["SimmSpline"]["y"]
                else:
                    x_values = t["function"]["MultiplierFunction"]["function"]["NaturalCubicSpline"]["x"]
                    y_values = t["function"]["MultiplierFunction"]["function"]["NaturalCubicSpline"]["y"]

                # Convert into numpy arrays
                x_values = np.array(x_values.split(), dtype=float)
                y_values = np.array(y_values.split(), dtype=float)

                assert len(x_values) > 1 and len(y_values) > 1, "Not enough points, can't fit a spline"

                # Fit a linear / quadratic / cubic / quartic function
                fit = np.polynomial.polynomial.Polynomial.fit(x_values, y_values, min(4, len(x_values) - 1))

                # A simple check to see if the fit is alright
                y_fit = fit(x_values)
                assert r2_score(y_values, y_fit) > 0.5, "A bad approximation of the SimmSpline"

                # Get the weights
                polycoef = np.zeros((5,))
                polycoef[:fit.coef.shape[0]] = fit.convert().coef

                # Update name; since this is a dependent joint variable the independent joint variable might already
                # have this name
                if params["name"] == params["original_name"]:
                    params["name"] = "{}_{}".format(params["name"], t["@name"])
                params["limited"] = True

                # Get min and max values
                y_fit = fit(x_values)
                params["range"] = np.array([min(y_fit), max(y_fit)])

                # Add a joint constraint between this joint and the independent joint, which we assume to be named
                # t["coordinates"]
                independent_joint = t["coordinates"]

                # Some dependent joint values may be coupled to another joint values. We need to find the name of
                # the independent joint
                # TODO We could do this after the model has been built since we just swap joint names,
                # then we wouldn't need to pass constraints into body/joint parser
                # params["motion_type"] is typically "coupled" for dependent joints, but not always, so let's just loop
                # through constraints and check
                #constraint_found = False

                # Go through all joint equality constraints
                for c in constraints["joint"]:
                    if c["@joint1"] != t["coordinates"]:
                        continue
                    else:
                        #constraint_found = True

                        # Check if this constraint is active
                        if c["@active"] != "true":
                            break

                        # Change the name of the independent joint
                        independent_joint = c["@joint2"]

                        # We're handling only an identity transformation for now
                        coeffs = np.array(c["@polycoef"].split(), dtype=float)
                        assert np.allclose(coeffs, np.array([0, 1, 0, 0, 0])), \
                            "We're handling only identity transformations for now"

                        break

                #assert constraint_found, "Couldn't find an independent joint for a coupled joint"

                #else:
                # Update motion type to dependent for posterity
                params["motion_type"] = "dependent"

                # These joint equality constraints don't seem to work properly. Is it because they're soft constraints?
                # E.g. the translations between femur and tibia should be strictly defined by knee angle, but it seems
                # like they're affected by gravity as well (tibia drops to translation range limit value when
                # leg6dof9musc is hanging from air) -> seems to work when solimp limits are very tight
                params["add_to_mujoco_joints"] = True

                # Add the equality constraint
                if params["add_to_mujoco_joints"]:
                    # We don't want to create a transform so set transform_value to zero
                    params["transform_value"] = 0
                    self.equality_constraints["joint"].append({"@name": params["name"] + "_constraint",
                                                               "@active": "true", "@joint1": params["name"],
                                                               "@joint2": independent_joint,
                                                               "@polycoef": Utils.array_to_string(polycoef),
                                                               "@solimp": "0.9999 0.9999 0.001 0.5 2"})

            elif Utils.is_nested_field(t, "LinearFunction", ["function"]):

                # I'm not sure how to handle a LinearFunction with coefficients != [1, 0] (the first one is slope,
                # second intercept), except for [-1, 0] when we can just flip the axis
                coefficients = np.array(t["function"]["LinearFunction"]["coefficients"].split(), dtype=float)
                assert abs(coefficients[0]) == 1 and coefficients[1] == 0, "How do we handle this linear function?"

                # If first coefficient is negative, flip the joint axis
                if coefficients[0] < 0:
                    flip_axis = True

                # Don't use transform_value here; we just want to use this joint as a mujoco joint
                # NOTE! We do need the transform_value for weld constraint if this joint is locked
                if "locked" in params and params["locked"]:
                    params["default_value_for_locked"] = params["transform_value"]
                params["transform_value"] = 0

            # Other functions are not defined yet
            else:
                print("Skipping transformation:")
                print(t)

            # Calculate new axis
            axis = np.array(t["axis"].split(), dtype=float)
            new_axis = np.matmul(self.orientation.transformation_matrix, Utils.create_transformation_matrix(axis))[:3, 3]
            params["axis"] = new_axis
            if flip_axis:
                params["axis"] *= -1

            # Figure out whether this is rotation or translation
            if t["@name"].startswith('rotation'):
                params["type"] = "hinge"
            elif t["@name"].startswith('translation'):
                params["type"] = "slide"
            else:
                raise TypeError("Unidentified transformation {}".format(t["@name"]))

            # If we add this joint then need to update T
            if params["transform_value"] != 0:
                if params["type"] == "hinge":
                    T_t = Utils.create_rotation_matrix(params["axis"], params["transform_value"])
                else:
                    T_t = Utils.create_translation_matrix(params["axis"], params["transform_value"])
                T = np.matmul(T, T_t)

            # Check if this joint/transformation should be added to mujoco_joints
            if params["add_to_mujoco_joints"]:
                self.mujoco_joints.append(params)

                # We might need this coordinate later for setting equality constraints between joints
                self.coordinates[t["coordinates"]] = params

            # We need to add an equality constraint for locked joints
            if "locked" in params and params["locked"]:

                # Create the constraint
                polycoef = np.array([params["default_value_for_locked"], 0, 0, 0, 0])
                constraint = {"@name": params["name"] + "_constraint", "@active": "true",
                              "@joint1": params["name"],
                              "@polycoef": Utils.array_to_string(polycoef)}

                # Add to equality constraints
                self.equality_constraints["joint"].append(constraint)

        return T

    @staticmethod
    def parse_coordinate_set(joint):
        # Parse all Coordinates defined for this joint

        coordinate_set = OrderedDict()
        if Utils.is_nested_field(joint, "Coordinate", ["CoordinateSet", "objects"]):
            coordinate = joint["CoordinateSet"]["objects"]["Coordinate"]

            # Make sure coordinate is a list
            if isinstance(coordinate, dict):
                coordinate = [coordinate]

            # Parse all Coordinates
            for c in coordinate:
                coordinate_set[c["@name"]] = {
                    "motion_type": c["motion_type"], "name": c["@name"],
                    "range": np.array(c["range"].split(), dtype=float),
                    "limited": True if c["clamped"] == "true" else False,
                    "locked": True if c["locked"] == "true" else False,
                    "transform_value": float(c["default_value"]) if "default_value" in c else None}

        return coordinate_set

    def parse_pin_joint(self, joint):

        # Start by parsing the CoordinateSet
        coordinate_set = self.parse_coordinate_set(joint)

        # There should be one coordinate for this joint
        assert len(coordinate_set.keys()) == 1, "There should be only one Coordinate for a PinJoint"
        params = copy.deepcopy(coordinate_set[next(iter(coordinate_set))])

        # Set default reference position/angle to zero. If this value is not zero, then you need
        # more care while calculating quartic functions for equality constraints
        params["ref"] = 0

        # We know this is a hinge joint; calculate new axis
        params["type"] = "hinge"
        axis = np.array([0, 0, 1])
        new_axis = np.matmul(self.orientation.transformation_matrix, Utils.create_transformation_matrix(axis))[:3, 3]
        params["axis"] = new_axis

        # Don't use transform_value here; we just want to use this joint as a mujoco joint
        # NOTE! We do need the transform_value for weld constraint if this joint is locked
        if "locked" in params and params["locked"]:
            params["default_value_for_locked"] = params["transform_value"]
        params["transform_value"] = 0

        # Append to mujoco joints
        self.mujoco_joints.append(params)

        # We need to add an equality constraint for locked joints
        if "locked" in params and params["locked"]:

            # Create the constraint
            polycoef = np.array([params["default_value_for_locked"], 0, 0, 0, 0])
            constraint = {"@name": params["name"] + "_constraint", "@active": "true",
                          "@joint1": params["name"],
                          "@polycoef": Utils.array_to_string(polycoef)}

            # Add to equality constraints
            self.equality_constraints["joint"].append(constraint)

    def parse_universal_joint(self, joint):

        # Start by parsing the CoordinateSet
        coordinate_set = self.parse_coordinate_set(joint)

        # There should be two coordinates for this joint
        assert len(coordinate_set.keys()) == 2, "There should be two Coordinates for a UniversalJoint"

        first = True
        for coordinate in coordinate_set:
            params = copy.deepcopy(coordinate_set[coordinate])

            # Set default reference position/angle to zero. If this value is not zero, then you need
            # more care while calculating quartic functions for equality constraints
            params["ref"] = 0

            # Both DoFs should be rotational, calculate new axes
            # TODO not sure if this is the correct order / axis
            assert params["motion_type"] == "rotational", "Both DoFs of an UniversalJoint should be rotational"
            params["type"] = "hinge"
            if first:
                axis = np.array([1, 0, 0])
                first = False
            else:
                axis = np.array([0, 1, 0])
            new_axis = np.matmul(self.orientation.transformation_matrix, Utils.create_transformation_matrix(axis))[:3, 3]
            params["axis"] = new_axis

            # Don't use transform_value here; we just want to use this joint as a mujoco joint
            # NOTE! We do need the transform_value for weld constraint if this joint is locked
            if "locked" in params and params["locked"]:
                params["default_value_for_locked"] = params["transform_value"]
            params["transform_value"] = 0

            # Append to mujoco joints
            self.mujoco_joints.append(params)

            # We need to add an equality constraint for locked joints
            if "locked" in params and params["locked"]:

                # Create the constraint
                polycoef = np.array([params["default_value_for_locked"], 0, 0, 0, 0])
                constraint = {"@name": params["name"] + "_constraint", "@active": "true",
                              "@joint1": params["name"],
                              "@polycoef": Utils.array_to_string(polycoef)}

                # Add to equality constraints
                self.equality_constraints["joint"].append(constraint)

    def get_equality_constraints(self, constraint_type):
        return self.equality_constraints[constraint_type]

    def get_coordinates(self):
        return self.coordinates


class Body:

    def __init__(self, obj):

        # Initialise parameters
        self.sites = []

        # Get important attributes
        self.name = obj["@name"]
        self.mass = float(obj["mass"])
        self.mass_center = np.array(obj["mass_center"].split(), dtype=float)
        self.inertia = np.array([obj[x] for x in
                                ["inertia_xx", "inertia_yy", "inertia_zz",
                                 "inertia_xy", "inertia_xz", "inertia_yz"]], dtype=float)

        # Get meshes if there are VisibleObjects
        self.mesh = []
        if "VisibleObject" in obj:

            # Get scaling of VisibleObject
            visible_object_scale = np.array(obj["VisibleObject"]["scale_factors"].split(), dtype=float)

            # There must be either "GeometrySet" or "geometry_files"
            if "GeometrySet" in obj["VisibleObject"] \
                    and obj["VisibleObject"]["GeometrySet"]["objects"] is not None:

                # Get mesh / list of meshes
                geometry = obj["VisibleObject"]["GeometrySet"]["objects"]["DisplayGeometry"]

                if isinstance(geometry, dict):
                    geometry = [geometry]

                for g in geometry:
                    display_geometry_scale = np.array(g["scale_factors"].split(), dtype=float)
                    total_scale = visible_object_scale * display_geometry_scale
                    g["scale_factors"] = Utils.array_to_string(total_scale)
                    self.mesh.append(g)

            elif "geometry_files" in obj["VisibleObject"] \
                    and obj["VisibleObject"]["geometry_files"] is not None:

                # Get all geometry files
                files = obj["VisibleObject"]["geometry_files"].split()
                for f in files:
                    self.mesh.append(
                        {"geometry_file": f,
                         "scale_factors": Utils.array_to_string(visible_object_scale)})

            else:
                print("No geometry files for body [{}]".format(self.name))

    def add_sites(self, path_point):
        for point in path_point:
            self.sites.append({"@name": point["@name"], "@pos": point["location"]})


class Muscle:

    def __init__(self, obj, muscle_type):
        # Note: Muscle class represents other types of actuators (just CoordinateActuator at the moment) as well
        self.muscle_type = muscle_type
        if muscle_type in ["CoordinateActuator", "PointActuator", "TorqueActuator"]:
            self.is_muscle = False
        else:
            self.is_muscle = True

        # Get important attributes
        self.name = obj["@name"]
        self.disabled = False if "isDisabled" not in obj or obj["isDisabled"] == "false" else True

        # The default range is from https://web.ecs.baylor.edu/faculty/garner/Research/GarnerPandy2003ParamEst.pdf
        self.range = [0.5, 1.5]

        # Parse time constants
        activation1 = np.nan
        activation2 = np.nan
        if "activation_time_constant" in obj:
            activation1 = obj["activation_time_constant"]
        elif "activation1" in obj:
            activation1 = obj["activation1"]
        if "deactivation_time_constant" in obj:
            activation2 = obj["deactivation_time_constant"]
        elif "activation2" in obj:
            activation2 = obj["activation2"]
        if "MuscleFirstOrderActivationDynamicModel" in obj:
            if "activation_time_constant" in obj["MuscleFirstOrderActivationDynamicModel"]:
                activation1 = obj["MuscleFirstOrderActivationDynamicModel"]["activation_time_constant"]
            if "deactivation_time_constant" in obj["MuscleFirstOrderActivationDynamicModel"]:
                activation2 = obj["MuscleFirstOrderActivationDynamicModel"]["deactivation_time_constant"]
        activation1 = np.float(activation1)
        activation2 = np.float(activation2)

        # Get time scale
        time_scale = 1.0
        if "time_scale" in obj:
            time_scale = np.float(obj["time_scale"])

        # This gives odd values some for models, let's just do it if time_scale is non-zero
        if time_scale != 1.0:
            # Linearize at act=0.5 ctrl=0.5 and calculate activation and deactivation time constants
            # Time constant calculations a la Florian Fischer
            act_linearization = ctrl_linearization = 0.5
            time_act = time_scale / ((0.5 + 1.5 * act_linearization) * (activation1 * ctrl_linearization + activation2))
            time_deact = time_scale * (0.5 + 1.5 * act_linearization) / activation2
            self.timeconst = np.array([time_act, time_deact])
        else:
            self.timeconst = np.array([activation1, activation2])

        # Get optimal fiber length, tendon slack length, and pennation angle
        self.optimal_fiber_length = np.float(obj.get("optimal_fiber_length", np.nan))
        self.tendon_slack_length = np.float(obj.get("tendon_slack_length", np.nan))
        self.pennation_angle = np.float(obj.get("pennation_angle_at_optimal", np.nan))

        # Estimate length range if optimal fiber length and tendon slack length are defined
        if np.all(np.isfinite([self.optimal_fiber_length, self.tendon_slack_length])):
            # Length range computations a la Florian Fischer
            self.length_range = np.array([0.5, 2]) * self.tendon_slack_length

            # Estimate actuator length ranges by minimizing error between mujoco and opensim optimal fiber length
            sol = minimize(mujoco_LO_loss, self.length_range,
                           args=(self.range, self.optimal_fiber_length, self.tendon_slack_length, self.pennation_angle))
            if sol.success:
                self.length_range = sol.x
        else:
            self.length_range = np.array([np.nan, np.nan])

        # Get damping for tendon -- not sure what the unit in OpenSim is, or how it relates to MuJoCo damping parameter
        self.tendon_damping = obj.get("damping", None)

        # Let's use max isometric force as an approximation for peak active force at rest
        self.force = obj.get("max_isometric_force", None)

        # Parse control limits
        self.limit = np.ones((2, 1))
        self.limit.fill(np.nan)
        if "min_control" in obj:
            self.limit[0] = obj["min_control"]
        if "max_control" in obj:
            self.limit[1] = obj["max_control"]

        # Get optimal force if defined (for non-muscle actuators only?)
        self.optimal_force = obj.get("optimal_force", None)

        # Get coordinate on which this actuator works (for non-muscle actuators only)
        self.coordinate = obj.get("coordinate", None)

        # Get path points so we can later add them into bodies; note that we treat
        # each path point type (i.e. PathPoint, ConditionalPathPoint, MovingPathPoint)
        # as a fixed path point; also note that non-muscle actuators don't have GeometryPaths
        if self.is_muscle:
            self.path_point_set = dict()
            self.sites = []
            path_point_set = obj["GeometryPath"]["PathPointSet"]["objects"]
            for pp_type in path_point_set:

                # TODO We're defining MovingPathPoints as fixed PathPoints and ignoring ConditionalPathPoints

                # Put the dict into a list of it's not already
                if isinstance(path_point_set[pp_type], dict):
                    path_point_set[pp_type] = [path_point_set[pp_type]]

                # Go through all path points
                for path_point in path_point_set[pp_type]:
                    if path_point["body"] not in self.path_point_set:
                        self.path_point_set[path_point["body"]] = []

                    if pp_type == "PathPoint":

                        # A normal PathPoint, easy to define
                        self.path_point_set[path_point["body"]].append(path_point)
                        self.sites.append({"@site": path_point["@name"]})

                    elif pp_type == "ConditionalPathPoint":

                        # Treat this is a fixed PathPoint, not kosher
                        print("Approximating a ConditionalPathPoint with a fixed PathPoint")
                        self.path_point_set[path_point["body"]].append(path_point)
                        self.sites.append({"@site": path_point["@name"]})

                    elif pp_type == "MovingPathPoint":

                        # We treat this as a fixed PathPoint, definitely not kosher
                        print("Approximating a MovingPathPoint with a fixed PathPoint")

                        # Get path point location
                        if "location" not in path_point:
                            location = np.array([0, 0, 0], dtype=float)
                        else:
                            location = np.array(path_point["location"].split(), dtype=float)

                        # Transform x,y, and z values (if they are defined) to their mean values to minimise error
                        location[0] = self.update_moving_path_point_location("x_location", path_point)
                        location[1] = self.update_moving_path_point_location("y_location", path_point)
                        location[2] = self.update_moving_path_point_location("z_location", path_point)

                        # Save the new location and the path point
                        path_point["location"] = Utils.array_to_string(location)
                        self.path_point_set[path_point["body"]].append(path_point)

                        self.sites.append({"@site": path_point["@name"]})

                    else:
                        raise TypeError("Undefined path point type {}".format(pp_type))

            # Finally, we need to sort the sites so that they are in correct order. Unfortunately we have to rely
            # on the site names since xmltodict decomposes the list into dictionaries. There's a pull request in
            # xmltodict for ordering children that might be helpful, but it has not been merged yet

            # Check that the site name prefixes are similar, and only the number is changing
            site_names = [d["@site"] for d in self.sites]
            prefix = os.path.commonprefix(site_names)
            try:
                numbers = [int(name[len(prefix):]) for name in site_names]
            except ValueError:
                raise ValueError("Check these site names, they might not be sorted correctly")

            self.sites = natsorted(self.sites, key=itemgetter(*['@site']), alg=ns.IGNORECASE)

    def update_moving_path_point_location(self, coordinate_name, path_point):
        if coordinate_name in path_point:
            # Parse x and y values
            if "SimmSpline" in path_point[coordinate_name]:
                x_values = np.array(path_point[coordinate_name]["SimmSpline"]["x"].split(), dtype=float)
                y_values = np.array(path_point[coordinate_name]["SimmSpline"]["y"].split(), dtype=float)
                pp_type = "spline"
            elif "MultiplierFunction" in path_point[coordinate_name]:
                x_values = np.array(path_point[coordinate_name]["MultiplierFunction"]["function"]["SimmSpline"]["x"].split(), dtype=float)
                y_values = np.array(path_point[coordinate_name]["MultiplierFunction"]["function"]["SimmSpline"]["y"].split(), dtype=float)
                pp_type = "spline"
            elif "NaturalCubicSpline" in path_point[coordinate_name]:
                x_values = np.array(path_point[coordinate_name]["NaturalCubicSpline"]["x"].split(), dtype=float)
                y_values = np.array(path_point[coordinate_name]["NaturalCubicSpline"]["y"].split(), dtype=float)
                pp_type = "spline"
            elif "PiecewiseLinearFunction" in path_point[coordinate_name]:
                x_values = np.array(path_point[coordinate_name]["PiecewiseLinearFunction"]["x"].split(), dtype=float)
                y_values = np.array(path_point[coordinate_name]["PiecewiseLinearFunction"]["y"].split(), dtype=float)
                pp_type = "piecewise_linear"
            else:
                raise NotImplementedError

            # Fit a cubic spline (if more than 2 values and pp_type is spline), otherwise fit a piecewise linear line
            if len(x_values) > 3 and pp_type == "spline":
                mdl = interp1d(x_values, y_values, kind="cubic", fill_value="extrapolate")
            else:
                mdl = interp1d(x_values, y_values, kind="linear", fill_value="extrapolate")

            # Return the mean of fit inside given range
            x = np.linspace(x_values[0], x_values[-1], 1000)
            return np.mean(mdl(x))

    def get_tendon(self):
        # Return MuJoCo tendon representation of this muscle
        tendon = {"@name": self.name + "_tendon", "site": self.sites}
        #if self.tendon_slack_length is not None:
        #    tendon["@springlength"] = self.tendon_slack_length
        #if self.tendon_damping is not None:
        #    tendon["@damping"] = self.tendon_damping
        return tendon

    def get_actuator(self):
        # Return MuJoCo actuator representation of this actuator
        actuator = {"@name": self.name}
        if self.is_muscle:
            actuator["@tendon"] = self.name + "_tendon"
            actuator["@class"] = "muscle"

            # Set timeconst
            if np.all(np.isfinite(self.timeconst)):
                actuator["@timeconst"] = Utils.array_to_string(self.timeconst)

            # Set max peak force at rest
            if self.force is not None:
                actuator["@force"] = self.force

            # Set estimated actuator length ranges
            if np.all(np.isfinite(self.length_range)):
                actuator["@lengthrange"] = Utils.array_to_string(self.length_range)

        else:
            #actuator["@gear"] = self.optimal_force
            actuator["@joint"] = self.coordinate
            actuator["@class"] = "motor"

        # Set ctrl limit
        if np.all(np.isfinite(self.limit)):
            actuator["@ctrllimited"] = "true"
            actuator["@ctrlrange"] = Utils.array_to_string(self.limit)

        return actuator

    def is_disabled(self):
        return self.disabled

def mujoco_LO_loss(length_range, range, optimal_fiber_length, tendon_slack_length, pennation_angle):
    """
    Computes squared Euclidean distance between MuJoCo and OpenSim model,
    regarding both optimal fiber length and constant tendon length/tendon slack length.

    Original code for this function was provided by Florian Fischer (2022)

    :param length_range: array of MuJoCo tendon length (=complete actuator length) ranges
    :param range: Operating length of muscle
    :param optimal_fiber_length: OpenSim optimal fiber length
    :param tendon_slack_length: OpenSim tendon slack length (or any reasonable constant tendon lengths)
    :param pennation_angle: OpenSim pennation angle at optimum
            (i.e., angle between tendon and fibers at optimal fiber length expressed in radians)
    :param use_optPennationAngle: Boolean; if this set to True, MuJoCo optimal fiber lengths LO should match
            OpenSim optimal fiber lengths LO_osim * cos(OpenSim pennation angle at optimum); otherwise, LO should match LO_osim
    :return: squared (unweighted) Euclidean distance of optimal fiber length and constant tendon lengths between MuJoCo and OpenSim
    """
    LO = estimate_fiber_length(length_range, range)
    LT = estimate_tendon_slack_length(length_range, range)

    if np.isnan(pennation_angle):
        pennation_angle = 0

    return np.linalg.norm(LO - optimal_fiber_length * np.cos(pennation_angle)) ** 2 + np.linalg.norm(LT - tendon_slack_length) ** 2

def estimate_fiber_length(length_range, range):
    return (length_range[0] - length_range[1]) / (range[0] - range[1])

def estimate_tendon_slack_length(length_range, range):
    return length_range[0] - range[0] * estimate_fiber_length(length_range, range)

def main(argv):
    converter = Converter()
    geometry_folder = None
    for_testing = False
    if len(argv) > 3:
        geometry_folder = argv[3]
    if len(argv) > 4:
        for_testing = True if argv[4].lower() == "true" else False
    converter.convert(argv[1], argv[2], geometry_folder, for_testing)


if __name__ == "__main__":
    main(sys.argv)
