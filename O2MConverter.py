import xmltodict
import numpy as np
import vtk
import admesh
import sys
import os
from contextlib import contextmanager


class Converter:
    """A class to convert OpenSim XML model files to MuJoCo XML model files"""

    def __init__(self):

        # Define input XML and output folder
        self.input_xml = None
        self.output_folder = None

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
            j = Joint(obj)

            # Add b to bodies
            self.bodies[b.name] = b

            # Ignore joint if it is None
            if j.parent_body is None:
                continue

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
        model["mujoco"]["compiler"] = {"@inertiafromgeom": "true", "@angle": "radian"}
        model["mujoco"]["default"] = {
            "joint": {"@limited": "true", "@damping": "1", "@armature": "0.01"},
            "geom": {"@contype": "1", "@conaffinity": "1", "@condim": "1", "@rgba": "0.8 0.6 .4 1",
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

        # Add sites to worldbody / "ground" in OpenSim
        worldbody["site"] = self.bodies[self.origin_joint.parent_body].sites

        # Add some more defaults
        worldbody["body"] = {
            "light": {"@mode": "trackcom", "@directional": "false", "@diffuse": ".8 .8 .8",
                      "@specular": "0.3 0.3 0.3", "@pos": "0 0 4.0", "@dir": "0 0 -1"},
            "joint": {
                "@name": "root", "@type": "free", "@pos": "0 0 0", "@limited": "false",
                "@damping": "0", "@armature": "0", "@stiffness": "0"}}

        # Build the kinematic chains
        worldbody["body"] = self.add_body(worldbody["body"], self.origin_body,
                                          self.joints[self.origin_body.name])

        # Add worldbody to the model
        model["mujoco"]["worldbody"] = worldbody

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

        return model

    def add_body(self, worldbody, current_body, current_joints):

        # Create a new MuJoCo body
        worldbody["@name"] = current_body.name

        # We need to find this body's position relative to parent body:
        # since we're progressing down the kinematic chain, each body
        # should have a joint to parent body
        joint_to_parent = self.find_joint_to_parent(current_body.name)
        worldbody["@pos"] = np.array2string(joint_to_parent.location_in_parent)[1:-1]

        # Add geom
        worldbody["geom"] = self.add_geom(current_body)

        # Add sites
        worldbody["site"] = current_body.sites

        # Add joints (except between origin body and it's parent, which
        # is typically "ground")
        if joint_to_parent.parent_body is not self.origin_joint.parent_body:
            joints = []
            for j in joint_to_parent.mujoco_joints:
                joints.append(
                    {"@name": j["name"], "@type": j["type"], "@pos": "0 0 0",
                     "@axis": np.array2string(j["axis"])[1:-1],
                     "@range": np.array2string(j["range"])[1:-1],
                     "@damping": "0", "@stiffness": "0", "@armature": "0.02"})
            if len(joints) > 0:
                worldbody["joint"] = joints

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

    def __init__(self, obj):

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

        # CustomJoint can represent any joint, we need to figure out
        # what kind of joint we're dealing with
        self.mujoco_joints = []
        if self.joint_type == "CustomJoint":
            self.parse_custom_joint(joint)
        elif self.joint_type == "WeldJoint":
            # Don't add anything to self.mujoco_joints, bodies are by default
            # attached rigidly to each other in MuJoCo
            pass
        else:
            print("Skipping a joint of type [{}]".format(self.joint_type))

        # And other parameters
        self.location_in_parent = np.array(joint["location_in_parent"].split(), dtype=float)
        self.orientation_in_parent = np.array(joint["orientation_in_parent"].split(), dtype=float)
        self.location = np.array(joint["location"].split(), dtype=float)
        self.orientation = np.array(joint["orientation"].split(), dtype=float)

    def parse_custom_joint(self, joint):
        # A CustomJoint in OpenSim model can represent any type of joint.
        # Try to parse the CustomJoint into a set of MuJoCo joints

        # Get transform axes
        transform_axes = joint["SpatialTransform"]["TransformAxis"]

        # Go through axes; they should be ordered so first there are rotations and then translations
        for t in transform_axes:

            # If there's a constant value or a SimmSpline, ignore this transformation
            if "Constant" in t["function"]:
                continue
            elif "MultiplierFunction" in t["function"]:
                if "Constant" in t["function"]["MultiplierFunction"]["function"] \
                        or "SimmSpline" in t["function"]["MultiplierFunction"]["function"]:
                    continue

            # Figure out whether this is rotation or translation
            params = dict()
            params["axis"] = np.array(t["axis"].split(), dtype=float)
            params["name"] = self.parent_body + "_" + self.child_body + "_" + t["@name"]

            # Find range from CoordinateSet
            coordinate = joint["CoordinateSet"]["objects"]["Coordinate"]
            if isinstance(coordinate, dict):
                coordinate = [coordinate]
            for c in coordinate:
                if c["@name"] == t["coordinates"]:
                    params["range"] = np.array(c["range"].split(), dtype=float)
                    break

            if t["@name"].startswith('rotation'):
                params["type"] = "hinge"
            elif t["@name"].startswith('translation'):
                params["type"] = "slide"
            else:
                raise "Unidentified transformation {}".format(t["@name"])

            # Add to set of joints
            self.mujoco_joints.append(params)


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
        actuator = {"@name": self.name + "_muscle", "@tendon": self.name + "_tendon"}
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