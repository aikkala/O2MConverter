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

import Utils


class Converter:
    """A class to convert OpenSim XML model files to MuJoCo XML model files"""

    def __init__(self):

        # Define input XML and output folder
        self.input_xml = None
        self.output_folder = None

        # List of constraints
        self.constraints = None

        # A OpenSim model consists of
        #  - a set of bodies
        #  - joints that connect those bodies
        #  - muscles that are connected to bodies
        self.bodies = dict()
        self.joints = dict()
        self.muscles = []

        # These dictionaries (or list of dicts) are in MuJoCo style (when converted to XML)
        self.asset = dict()
        self.tendon = []
        self.actuator = []
        self.equality = {"joint": []}

        self.geometry_folder = None
        self.output_geometry_folder = "Geometry/"
        self.vtk_reader = vtk.vtkXMLPolyDataReader()
        self.stl_writer = vtk.vtkSTLWriter()

        # Setup writer
        self.stl_writer.SetInputConnection(self.vtk_reader.GetOutputPort())
        self.stl_writer.SetFileTypeToBinary()

        self.origin_body = None
        self.origin_joint = None

    def convert(self, input_xml, output_folder, geometry_folder=None):
        """Convert given OpenSim XML model to MuJoCo XML model"""

        # Save input and output XML files in case we need them somewhere
        self.input_xml = input_xml

        # Set geometry folder
        self.geometry_folder = geometry_folder

        # Read input_xml and parse it
        with open(input_xml) as f:
            text = f.read()
        p = xmltodict.parse(text)

        # Set output folder
        model_name = p["OpenSimDocument"]["Model"]["@name"] + "_converted"
        self.output_folder = output_folder + "/" + model_name + "/"

        # Create the output folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Get constraints
        if "ConstraintSet" in p["OpenSimDocument"]["Model"]:
            self.constraints = p["OpenSimDocument"]["Model"]["ConstraintSet"]["objects"]

        # Find and parse bodies and joints
        if "BodySet" in p["OpenSimDocument"]["Model"]:
            self.parse_bodies_and_joints(p["OpenSimDocument"]["Model"]["BodySet"]["objects"])

        # Find and parse muscles
        if "ForceSet" in p["OpenSimDocument"]["Model"]:
            self.parse_muscles_and_tendons(p["OpenSimDocument"]["Model"]["ForceSet"]["objects"])

        # Now we need to re-assemble them in MuJoCo format
        # (or actually a dict version of the model so we can use
        # xmltodict to save the model into a XML file)
        mujoco_model = self.build_mujoco_model(model_name)

        # Finally, save the MuJoCo model into XML file
        output_xml = self.output_folder + model_name + ".xml"
        with open(output_xml, 'w') as f:
            f.write(xmltodict.unparse(mujoco_model, pretty=True))

        # We might need to fix stl files (if converted from OpenSim Geometry vtk files)
        if self.geometry_folder is not None:
            self.fix_stl_files()

    def parse_bodies_and_joints(self, p):

        # Go through all bodies and their joints
        for obj in p["Body"]:
            b = Body(obj)
            j = Joint(obj, self.constraints)

            # Add b to bodies
            self.bodies[b.name] = b

            # Ignore joint if it is None
            if j.parent_body is None:
                continue

            # Add joint equality constraints
            self.equality["joint"].extend(j.get_equality_constraints())

            # There might be multiple joints per body
            if j.parent_body not in self.joints:
                self.joints[j.parent_body] = []
            self.joints[j.parent_body].append(j)

    def parse_muscles_and_tendons(self, p):

        # Go through all muscle types (typically there are only one type of muscle)
        for muscle_type in p:

            # Skip some forces
            if muscle_type in ["HuntCrossleyForce", "CoordinateLimitForce"]:
                print("Skipping a force: {}".format(muscle_type))
                continue

            if isinstance(p[muscle_type], dict):
                p[muscle_type] = [p[muscle_type]]

            for muscle in p[muscle_type]:
                m = Muscle(muscle)
                self.muscles.append(m)
                if m.is_disabled():
                    continue
                else:
                    self.tendon.append(m.get_tendon())
                    self.actuator.append(m.get_actuator())

                    # Add sites to all bodies this muscle/tendon spans
                    for body_name in m.path_point_set:
                        self.bodies[body_name].add_sites(m.path_point_set[body_name])

    def build_mujoco_model(self, model_name):
        # Initialise model
        model = {"mujoco": {"@model": model_name}}

        # Set defaults
        model["mujoco"]["compiler"] = {"@inertiafromgeom": "auto", "@angle": "radian"}
        model["mujoco"]["default"] = {
            "joint": {"@limited": "true", "@damping": "1", "@armature": "0.01", "@stiffness": "5"},
            "geom": {"@contype": "1", "@conaffinity": "1", "@condim": "3", "@rgba": "0.8 0.6 .4 1",
                     "@margin": "0.001", "@solref": ".02 1", "@solimp": ".8 .8 .01", "@material": "geom"},
            "site": {"@size": "0.01"}}
        model["mujoco"]["option"] = {"@timestep": "0.002", "@iterations": "50", "@solver": "PGS",
                                     "flag": {"@energy": "enable"}}
        model["mujoco"]["size"] = {"@nconmax": "400"}
        model["mujoco"]["visual"] = {
            "map": {"@fogstart": "3", "@fogend": "5", "@force": "0.1"},
            "quality": {"@shadowsize": "2048"}}

        # Start building the worldbody
        worldbody = {"geom": {"@name": "floor", "@pos": "0 0 0", "@size": "10 10 0.125",
                              "@type": "plane", "@material": "MatPlane", "@condim": "3"}}

        # We should probably find the "origin" body, where the kinematic chain begins
        self.origin_body, self.origin_joint = self.find_origin()

        # Rotate self.origin_joint.orientation_in_parent so the model is upright
        self.origin_joint.orientation_in_parent = self.origin_joint.orientation_in_parent.rotate(
            Quaternion(axis=[1, 0, 0], angle=math.pi/2))

        # Increase z-coordinate also to make sure the whole thing is above floor
        self.origin_joint.location_in_parent[2] = self.origin_joint.location_in_parent[2] + 1.5

        # Add sites to worldbody / "ground" in OpenSim
        worldbody["site"] = self.bodies[self.origin_joint.parent_body].sites

        # Add some more defaults
        worldbody["body"] = {
            "light": {"@mode": "trackcom", "@directional": "false", "@diffuse": ".8 .8 .8",
                      "@specular": "0.3 0.3 0.3", "@pos": "0 0 4.0", "@dir": "0 0 -1"}}

        # Build the kinematic chains
        worldbody["body"] = self.add_body(worldbody["body"], self.origin_body,
                                          self.joints[self.origin_body.name])

        # Add worldbody to the model
        model["mujoco"]["worldbody"] = worldbody

        # We might want to use a weld constraint to fix the origin body to worldbody for experiments
        self.equality["weld"] = {"@name": "origin_to_worldbody", "@body1": self.origin_body.name, "@active": "false"}

        # Set some asset defaults
        self.asset["texture"] = [
            {"@name": "texplane", "@type": "2d", "@builtin": "checker", "@rgb1": ".2 .3 .4",
             "@rgb2": ".1 0.15 0.2", "@width": "100", "@height": "100"},
            {"@name": "texgeom", "@type": "cube", "@builtin": "flat", "@mark": "cross",
             "@width": "127", "@height": "1278", "@rgb1": "0.8 0.6 0.4", "@rgb2": "0.8 0.6 0.4",
             "@markrgb": "1 1 1", "@random": "0.01"}]

        self.asset["material"] = [
            {"@name": "MatPlane", "@reflectance": "0.5", "@texture": "texplane",
             "@texrepeat": "1 1", "@texuniform": "true"},
            {"@name": "geom", "@texture": "texgeom", "@texuniform": "true"}]

        # Add assets to model
        model["mujoco"]["asset"] = self.asset

        # Add tendons and actuators
        model["mujoco"]["tendon"] = {"spatial": self.tendon}
        model["mujoco"]["actuator"] = {"muscle": self.actuator}

        # Add equality constraints between joints
        model["mujoco"]["equality"] = self.equality

        return model

    def add_body(self, worldbody, current_body, current_joints):

        # Create a new MuJoCo body
        worldbody["@name"] = current_body.name

        # We need to find this body's position relative to parent body:
        # since we're progressing down the kinematic chain, each body
        # should have a joint to parent body
        joint_to_parent = self.find_joint_to_parent(current_body.name)

        # Define position and orientation
        worldbody["@pos"] = np.array2string(joint_to_parent.location_in_parent)[1:-1]
        worldbody["@quat"] = "{} {} {} {}"\
            .format(joint_to_parent.orientation_in_parent.w,
                    joint_to_parent.orientation_in_parent.x,
                    joint_to_parent.orientation_in_parent.y,
                    joint_to_parent.orientation_in_parent.z)


        # Add inertial properties -- only if mass is greater than zero and eigenvalues are positive
        # (if "inertial" is missing MuJoCo will infer the inertial properties from geom)
        if current_body.mass > 0:
            values, vectors = np.linalg.eig(Utils.create_symmetric_matrix(current_body.inertia))
            if np.all(values > 0):
                worldbody["inertial"] = {"@pos": np.array2string(current_body.mass_center)[1:-1],
                                         "@mass": str(current_body.mass),
                                         "@fullinertia": np.array2string(current_body.inertia)[1:-1]}

        # Add geom
        worldbody["geom"] = self.add_geom(current_body)

        # Add sites
        worldbody["site"] = current_body.sites

        # Go through joints
        worldbody["joint"] = []
        for mujoco_joint in joint_to_parent.mujoco_joints:

            # Define the joint
            j = {"@name": mujoco_joint["name"], "@type": mujoco_joint["type"], "@pos": "0 0 0",
                 "@limited": "true" if mujoco_joint["limited"] else "false",
                 "@axis": np.array2string(mujoco_joint["axis"])[1:-1],
                 "@range": np.array2string(mujoco_joint["range"])[1:-1]}

            # If the joint is between origin body and it's parent, which should be "ground", set
            # damping, armature, and stiffness to zero
            if joint_to_parent is self.origin_joint:
                j.update({"@armature": 0, "@damping": 0, "@stiffness": 0})

            # Add to joints
            worldbody["joint"].append(j)

        #if len(joints) > 0:
        #    worldbody["joint"] = joints

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
                         "@size": np.array2string(size)[1:-1]})

        else:

            # Make sure output geometry folder exists
            os.makedirs(self.output_folder + self.output_geometry_folder, exist_ok=True)

            # Grab the mesh from given geometry folder
            for m in body.mesh:

                # Get file path
                vtk_file = self.geometry_folder + "/" + m["geometry_file"]

                # Check the file exists
                if not os.path.exists(vtk_file) or not os.path.isfile(vtk_file):
                    raise "Mesh file doesn't exist"

                # Transform the vtk file into an stl file and save it
                mesh_name = m["geometry_file"][:-4]
                stl_file = self.output_geometry_folder + mesh_name + ".stl"
                self.vtk_reader.SetFileName(vtk_file)
                self.stl_writer.SetFileName(self.output_folder + stl_file)
                self.stl_writer.Write()

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
        if len(self.joints) == 0:
            raise "There are no joints!"

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

        if joint_to_parent is None:
            raise "Couldn't find joint to parent body for body " + body_name

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

        # This code assumes there's max one joint per object
        joint = obj["Joint"]
        self.parent_body = None

        # 'ground' body does not have joints
        if joint is None or len(joint) == 0:
            return
        elif len(joint) > 1:
            raise 'Multiple joints for one body'

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
        self.orientation = np.array(joint["orientation"].split(), dtype=float)

        # TODO we assume orientation_in_parent is [0, 0, 0]
        self.orientation_in_parent = np.array(joint["orientation_in_parent"].split(), dtype=float)
        if np.sum(np.abs(self.orientation_in_parent)) > 1e-6:
            raise "We're not doing this correctly"
        self.orientation_in_parent = Quaternion(1, 0, 0, 0)

        # Some joint values are dependent on other joint values; we need to create equality constraints between those
        self.equality_constraints = []

        # CustomJoint can represent any joint, we need to figure out
        # what kind of joint we're dealing with
        self.mujoco_joints = []
        if self.joint_type == "CustomJoint":
            T = self.parse_custom_joint(joint, constraints)
            T_location_in_parent = np.eye(4, 4)
            #T_location_in_parent[:3, 3] = self.location_in_parent
            T = np.matmul(T_location_in_parent, T)
            # TODO FIX THIS! location_in_parent should be updated but not exactly overwritten
            self.orientation_in_parent = Quaternion(matrix=T)
            self.location_in_parent = self.location_in_parent + T[:3, 3]

        elif self.joint_type == "WeldJoint":
            # Don't add anything to self.mujoco_joints, bodies are by default
            # attached rigidly to each other in MuJoCo
            pass
        else:
            print("Skipping a joint of type [{}]".format(self.joint_type))

    def parse_custom_joint(self, joint, constraints):
        # A CustomJoint in OpenSim model can represent any type of joint.
        # Try to parse the CustomJoint into a set of MuJoCo joints

        # Get transform axes
        transform_axes = joint["SpatialTransform"]["TransformAxis"]

        # We might need to create a homogeneous transformation matrix from
        # location_in_parent to actual joint location
        T = np.eye(4, 4)

        # Start by parsing the CoordinateSet
        coordinate_set = dict()
        if Utils.is_nested_field(joint, "Coordinate", ["CoordinateSet", "objects"]):
            coordinate = joint["CoordinateSet"]["objects"]["Coordinate"]

            # Make sure coordinate is a list
            if isinstance(coordinate, dict):
                coordinate = [coordinate]

            # Parse all Coordinates
            for c in coordinate:
                if c["locked"] == "true":
                    raise "We aren't handling this situation; means that joint is locked to the default value"

                coordinate_set[c["@name"]] = {
                    "motion_type": c["motion_type"], "name": c["@name"],
                    "range": np.array(c["range"].split(), dtype=float),
                    "limited": True if c["clamped"] == "true" else False}

        # Go through axes; rotations should come first and translations after
        order = ["rotation1", "rotation2", "rotation3", "translation1", "translation2", "translation3"]
        for t in transform_axes:

            # Make sure order is correct
            if order[0] != t["@name"]:
                raise "Wrong order of transformations"
            else:
                order.pop(0)

            # Use the Coordinate parameters we parsed earlier; note that these do not exist for all joints (e.g
            # constant joints)
            if t["coordinates"] in coordinate_set:
                params = copy.deepcopy(coordinate_set[t["coordinates"]])
            else:
                params = {"name": "{}_{}".format(joint["@name"], t["@name"])}

            # Set default reference position/angle to zero. These probably need to be zero for joint equality
            # constraints (i.e. the quartic functions)!
            params["ref"] = 0

            # By default don't treat this joint as a fixed joint (or update the joint location)
            params["transform_value"] = 0
            params["add_to_mujoco_joints"] = True

            # Handle a "Constant" transformation. We're not gonna create this joint
            # but we need the transformation information to properly align the joint
            if Utils.is_nested_field(t, "Constant", ["function"]) or \
                    Utils.is_nested_field(t, "Constant", ["function", "MultiplierFunction", "function"]):

                # Get the value
                if "MultiplierFunction" in t["function"]:
                    value = float(t["function"]["MultiplierFunction"]["function"]["Constant"]["value"])
                else:
                    value = float(t["function"]["Constant"]["value"])

                # If the value is near zero don't bother creating this joint
                if abs(value) < 1e-6:
                    continue

                # Otherwise define a limited MuJoCo joint (we're not really creating this (sub)joint, we just update the
                # joint position)
                params["limited"] = "true"
                params["range"] = np.array([value])
                params["transform_value"] = value
                params["add_to_mujoco_joints"] = False

            # Handle a "SimmSpline" or "NaturalCubicSpline" transformation with a quartic approximation
            elif Utils.is_nested_field(t, "SimmSpline", ["function", "MultiplierFunction", "function"]) or \
                    Utils.is_nested_field(t, "NaturalCubicSpline", ["function", "MultiplierFunction", "function"]) or \
                    Utils.is_nested_field(t, "SimmSpline", ["function"]) or \
                    Utils.is_nested_field(t, "NaturalCubicSpline", ["function"]):

                # We can't model the relationship between two joints using a spline, but we can try to approximate it
                # with a quartic function. So fit a quartic function and check that the error is small enough; if the
                # error is too large, use a constant value

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

                if len(x_values) < 1 or len(y_values) < 1:
                    raise "Not enough points"

                # Fit the spline (I'm not sure what kind of spline SimmSpline is, but let's use a cubic spline here)
                f_spline = interp1d(x_values, y_values, kind="cubic")

                # Fit the quartic approximation
                f_quartic = np.polynomial.polynomial.Polynomial.fit(x_values, y_values, 4)

                # Estimate error between these fits with an Rsquared
                fit_spline = f_spline(x_values)
                fit_quartic = f_quartic(x_values)
                ssreg = np.sum((fit_quartic - np.mean(fit_spline)) ** 2)
                sstot = np.sum((fit_spline - np.mean(fit_spline)) ** 2)
                if ssreg / sstot < 0.5:
                    raise "A bad quartic approximation of the cubic spline"

                # We need to do a transformation also (really only needed if f_quartic(0) is non-zero, but it's likely
                # to always be non-zero), and modify the quartic function
                params["transform_value"] = f_quartic(0)
                y_values = y_values - params["transform_value"]

                # Get the new quartic function
                f_quartic = np.polynomial.polynomial.Polynomial.fit(x_values, y_values, 4)

                # Get the weights
                polycoef = f_quartic.coef

                # Update name; since this is a dependent joint variable the independent joint variable might already
                # have this name
                params["name"] = "{}_{}".format(params["name"], t["@name"])

                # Whether or not this joint is limited is up to the independent joint
                params["limited"] = True

                # Get min and max values
                fit_quartic = f_quartic(x_values)
                params["range"] = np.array([min(fit_quartic), max(fit_quartic)])

                # Add a joint constraint between this joint and the independent joint, which we assume to be named
                # t["coordinates"]
                independent_joint = t["coordinates"]

                # Some dependent joint values may be coupled to another joint values. We need to find the name of
                # the independent joint
                if "motion_type" in params and params["motion_type"] == "coupled":

                    # Find the constraint that pertains to this joint
                    constraint_found = False
                    if constraints is not None and "CoordinateCouplerConstraint" in constraints:
                        # Make sure CoordinateCouplerConstraint is a list
                        if isinstance(constraints["CoordinateCouplerConstraint"], dict):
                            constraints["CoordinateCouplerConstraint"] = [constraints["CoordinateCouplerConstraint"]]

                        # Go through all CoordinateCouplerConstraints
                        for c in constraints["CoordinateCouplerConstraint"]:
                            if c["isDisabled"] == "true" or c["dependent_coordinate_name"] != t["coordinates"]:
                                continue
                            else:
                                constraint_found = True
                                # Change the name of the independent joint
                                independent_joint = c["independent_coordinate_names"]
                                if Utils.is_nested_field(c, "coefficients", ["coupled_coordinates_function", "LinearFunction"]):
                                    # Update params["polycoef"]
                                    coeffs = np.array(c["coupled_coordinates_function"]["LinearFunction"]["coefficients"].split(),
                                                       dtype=float)
                                    polycoef = coeffs[0]*polycoef
                                    polycoef[0] = polycoef[0] + coeffs[1]
                                break

                    if not constraint_found:
                        raise "Couldn't find an independent joint for a coupled joint"

                else:
                    # Update motion type to dependent for posterity
                    params["motion_type"] = "dependent"

                # Add the equality constraint
                self.equality_constraints.append({"@name": params["name"] + "_constraint", "@active": "true",
                                                  "@joint1": params["name"], "@joint2": independent_joint,
                                                  "@polycoef": np.array2string(polycoef, suppress_small=True)[1:-1]})

            elif Utils.is_nested_field(t, "LinearFunction", ["function"]):

                # I'm not sure how to handle a LinearFunction with coefficients != [1, 0] (the first one is slope,
                # second intercept)
                coefficients = np.array(t["function"]["LinearFunction"]["coefficients"].split(), dtype=float)
                if coefficients[0] != 1 or coefficients[1] != 0:
                    raise "How do we handle this linear function?"

            # Other functions are not defined yet
            else:
                print("Skipping transformation:")
                print(t)

            # Figure out whether this is rotation or translation
            params["axis"] = np.array(t["axis"].split(), dtype=float)

            if t["@name"].startswith('rotation'):
                params["type"] = "hinge"
            elif t["@name"].startswith('translation'):
                params["type"] = "slide"
            else:
                raise "Unidentified transformation {}".format(t["@name"])

            # If do_transform is True, then we need to update T
            if params["transform_value"] != 0:
                if params["type"] == "hinge":
                    T_t = Utils.create_rotation_matrix(params["axis"], params["transform_value"])
                else:
                    T_t = Utils.create_translation_matrix(params["axis"], params["transform_value"])
                T = np.matmul(T, T_t)

            # Check if this joint/transformation should be added to mujoco_joints
            if params["add_to_mujoco_joints"]:
                self.mujoco_joints.append(params)

        return T

    def get_equality_constraints(self):
        return self.equality_constraints


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
                    g["scale_factors"] = np.array2string(total_scale)[1:-1]
                    self.mesh.append(g)

            elif "geometry_files" in obj["VisibleObject"] \
                    and obj["VisibleObject"]["geometry_files"] is not None:

                # Get all geometry files
                files = obj["VisibleObject"]["geometry_files"].split()
                for f in files:
                    self.mesh.append(
                        {"geometry_file": f,
                         "scale_factors": np.array2string(visible_object_scale)[1:-1]})

            else:
                print("No geometry files for body [{}]".format(self.name))

    def add_sites(self, path_point):
        for point in path_point:
            self.sites.append({"@name": point["@name"], "@pos": point["location"]})


class Muscle:

    def __init__(self, obj):

        # Get important attributes
        self.name = obj["@name"]
        self.disabled = False if "isDisabled" not in obj or obj["isDisabled"] == "false" else True
        self.timeconst = np.asarray([obj["activation_time_constant"], obj["deactivation_time_constant"]], dtype=float)

        # Get path points so we can later add them into bodies; note that we treat
        # each path point type (i.e. PathPoint, ConditionalPathPoint, MovingPathPoint)
        # as a fixed path point
        self.path_point_set = dict()
        self.sites = []
        path_point_set = obj["GeometryPath"]["PathPointSet"]["objects"]
        for pp_type in path_point_set:

            # Put the dict into a list of it's not already
            if isinstance(path_point_set[pp_type], dict):
                path_point_set[pp_type] = [path_point_set[pp_type]]

            # Go through all path points
            for path_point in path_point_set[pp_type]:
                if path_point["body"] not in self.path_point_set:
                    self.path_point_set[path_point["body"]] = []
                self.path_point_set[path_point["body"]].append(path_point)
                self.sites.append({"@site": path_point["@name"]})

    def get_tendon(self):
        # Return MuJoCo tendon representation of this muscle
        tendon = {"@name": self.name + "_tendon", "@width": "0.004",
                  "@rgba": ".95 .3 .3 1", "@limited": "false", "site": self.sites}
        return tendon

    def get_actuator(self):
        # Return MuJoCo actuator representation of this muscle
        actuator = {"@name": self.name, "@tendon": self.name + "_tendon",
                    "@timeconst": np.array2string(self.timeconst)[1:-1]}
        return actuator

    def is_disabled(self):
        return self.disabled


if __name__ == "__main__":
    converter = Converter()
    if len(sys.argv) > 3:
        geometry_folder = sys.argv[3]
    else:
        geometry_folder = None
    converter.convert(sys.argv[1], sys.argv[2], geometry_folder)