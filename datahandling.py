import time
import numpy as np


std_inputs_dict = {'core_length': 20.0,
                   'core_minor_axis': 10.0,
                   'core_major_axis': 20.0,
                   'num_of_turns': 3.0,
                   'outer_spacing': 5.0,
                   'spacing': 0.5
                   }
yes_answer = ['y', 'Y', 'Yes', 'YES', 'yes']
no_answer = ['n', 'N', 'no', 'NO', 'No']
file_writes = {}
file_reads = {}


def input_data(input_dict=False):
    if not input_dict:
        input_dict = std_inputs_dict
    else:
        input_dict = std_inputs_dict
        for key in input_dict:
            input_dict[key] = float(input('Input ' + str(key) + ': '))
    return input_dict


def read_data(file):
    with open(file) as f:
        lines = f.readlines()
        input_dict = {}
        for line in lines:
            input_dict[line[0]] = line[1]
    return input_dict


def store_coil_points(array, filename='coil_array_points.bin'):
    f = open(filename, mode='w')
    f.write(array.tobytes())
    f.close()
    print(str(len(array.tobytes())) + ' bytes successfully written to ' + filename)
    file_writes[filename] = time.ctime()


def fetch_coil_points(filename='coil_array_points.bin'):
    array = np.fromfile(filename)
    file_reads[filename] = time.ctime()
    return array


def start_up():
    running = True
    while running:
        pass_manually = input('pass dimensions manually? (y/n): ')
        if pass_manually in yes_answer:
            input_dict = input_data(input_dict=True)
            running = False
            return input_dict
        elif pass_manually in no_answer:
            input_dict = std_inputs_dict
            # filename = input('enter filepath of coil dimensions: ')
            # input_dict = self.read_data(filename)
            running = False
            return input_dict
        else:
            pass
