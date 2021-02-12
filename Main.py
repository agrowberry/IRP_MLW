import plotly.graph_objects as go
import numpy as np
import sympy as sp


# define main winding dimension - call the iterative path and slice building fcns.
class Main:
    def __init__(self):
        while True:

            self.pass_manually = input('Pass Geometries Manually? (y/n): ')
            self.yes_answer = 'y' or 'Y' or 'Yes' or 'YES' or 'yes'
            self.no_answer = 'n' or 'N' or 'no' or 'NO' or 'No'


            if self.pass_manually == self.yes_answer:
                self.core_length = float(input('Input Core Length: '))
                self.core_minor_axis = float(input('Input Core Major Width: '))
                self.core_major_axis = float(input('Input Core Minor Width: '))
                break

            if self.pass_manually == self.no_answer:
                geom_file = input('Enter filepath: ')
                open(geom_file)
                break

            else:
                print('Unrecognised Input')
                pass

    def build_spiral(self, N):







spyro = Main()







