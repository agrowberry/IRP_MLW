import time
import numpy as np
import pandas as pd
import json
import sys
import pathlib
import open3d as o3d


current_directory = pathlib.Path().absolute()
# echo is conditional for whether or not to print outputs to console
echo = True
# standard set of dimensions
std_inputs_dict = {
    "core_length": 10.0,
    "core_minor_axis": 10.0,
    "core_major_axis": 20.0,
    "core_radius": 1.0,
    "coil_radius_percentage": 0.25,
    "num_of_turns": 8,
    "outer_spacing": 5.0,
    "spacing": 0.1,
}
yes_answer = ["y", "Y", "Yes", "YES", "yes"]
no_answer = ["n", "N", "no", "NO", "No"]
file_writes = {}
file_reads = {}


def input_data(input_dict=None):
    if input_dict is None:
        input_dict = std_inputs_dict
    else:
        input_dict = std_inputs_dict
        for key in input_dict:
            input_dict[key] = float(input("Input " + str(key) + ": "))
    return input_dict


def read_data(file):
    with open(file) as f:
        lines = f.readlines()
        input_dict = {}
        for line in lines:
            input_dict[line[0]] = line[1]
    return input_dict


def store_dict(dict, filename):
    with open(filename, "w") as outfile:
        json.dump(dict, outfile)
    print("dictionary stored at: %s" % filename)
    file_writes[filename] = time.ctime()


def read_dict(filename):
    with open(filename, "r") as openfile:
        stored_dict = json.load(openfile)
    print("dictionary read from: %s" % filename)
    file_reads[filename] = time.ctime()
    return stored_dict


def store_coil_points(array, filename="docs/coil_array_points.json"):
    df = pd.DataFrame(np.transpose(array), columns=["x", "y", "z"])
    df_json = df.to_json(orient="records")
    with open(filename, "w") as outfile:
        json.dump(df_json, outfile)
    if echo:
        print(
            str(sys.getsizeof(json.dumps(df_json)))
            + " bytes successfully written to %s/%s"
            % (str(current_directory), filename)
        )
    file_writes[filename] = time.ctime()


def fetch_coil_points(filename="docs/coil_array_points.json"):
    with open(filename, "r") as openfile:
        json_object = json.load(openfile)
    df = pd.read_json(json_object, orient="records")
    file_reads[filename] = time.ctime()
    if echo:
        print("points fetched successfully from %s/%s" % (current_directory, filename))
    return np.array([pd.array(df["x"]), pd.array(df["y"]), pd.array(df["z"])])


def write_mesh_stl(mesh, filename="docs/winding_coil.stl"):
    print("writing %s to %s/%s" % (mesh, str(current_directory), filename))
    o3d.io.write_triangle_mesh(filename, mesh)
    print(
        str(sys.getsizeof(filename))
        + " bytes successfully written to %s/%s" % (str(current_directory), filename)
    )


def store_complete_info(header_dict, array, normal_array, filename):
    array_to_store = array
    point_dict = {'x': list(array_to_store[0]), 'y': list(array_to_store[1]), 'z': list(array_to_store[2])}
    array_to_store = np.transpose(normal_array)
    normal_dict = {'x': list(array_to_store[0]), 'y': list(array_to_store[1]), 'z': list(array_to_store[2])}
    stored_dict = {'header': header_dict, 'points': point_dict, 'normals': normal_dict}
    with open(filename, "w") as outfile:
        json.dump(stored_dict, outfile)
    if echo:
        print(
            str(sys.getsizeof(stored_dict))
            + " bytes successfully written to %s/%s"
            % (str(current_directory), filename)
        )
    file_writes[filename] = time.ctime()


def fetch_complete_info(filename):
    with open(filename, "r") as openfile:
        stored_dict = json.load(openfile)
    file_reads[filename] = time.ctime()
    if echo:
        print("points fetched successfully from %s/%s" % (current_directory, filename))
    point_dict = stored_dict['points']
    points_array = np.array([np.array(point_dict['x']), np.array(point_dict['y']), np.array(point_dict['z'])])
    normal_dict = stored_dict['normals']
    normal_array = np.array([np.array(normal_dict['x']), np.array(normal_dict['y']), np.array(normal_dict['z'])])
    header_dict = stored_dict['header']
    return header_dict, points_array, normal_array


def start_up():
    running = True
    while running:
        use_std_inputs = input("use standard dimensions? (y/n): ")
        if use_std_inputs in no_answer:
            pass_manually = input("pass dimensions manually? (y/n): ")
            if pass_manually in yes_answer:
                input_dict = input_data(input_dict=True)
                running = False
            elif pass_manually in no_answer:
                filename = input("enter filepath of coil dimensions: ")
                input_dict = read_data(filename)
                running = False
            else:
                print("unrecognised input...")
                pass
        elif use_std_inputs in yes_answer:
            input_dict = std_inputs_dict
            running = False
        else:
            print("unrecognised input...")
            pass
    return input_dict


def progressbar(current, maximum, message="working", exit_message="finished"):
    if echo:
        if current == 0:
            print(message)
            progressbar.last_percentage = None
        percentage = int(round(((current + 1) / maximum) * 100))
        barfill = int(round(percentage * 0.5))
        lp = progressbar.last_percentage
        if current + 1 == maximum:
            print("[%-50s] %d%%" % ("=" * barfill, percentage))
            print(exit_message, end="\n")
        else:
            if percentage != progressbar.last_percentage:
                print("[%-50s] %d%%" % ("=" * barfill, percentage), end="\r")
                time.sleep(0.05)
        progressbar.last_percentage = percentage
    elif not echo:
        pass


def workingmessage(message):
    if echo:
        output_slashes = ["|", "/", "-", "\\", "|", "/", "-", "\\"]
        print(
            "%s : %1s ..." % (message, output_slashes[workingmessage.count % 7]),
            end="\r",
        )
        workingmessage.count += 1
        time.sleep(0.1)
    elif not echo:
        pass
