"""
powerflow.py

Started by Owen Frausto on April 19, 2025. 
Meant to contain classes for a larger power flow simulator built on top of numpy.

"""

from beartype import beartype, typing
from typing import Literal, List
import numpy as np

# Bus parent class
class Bus:
    def __init__(self, bus_id: int):
        self.id = bus_id

# Define bus subclasses
class PQBus(Bus):
    def __init__(self, id, Pd, Qd):
        super().__init__(id)
        self.Pd = Pd
        self.Qd = Qd

class PVBus(Bus):
    def __init__(self, id, Pg, Vspec):
        super().__init__(id)
        self.Pg = Pg
        self.Vspec = Vspec

class SlackBus(Bus):
    def __init__(self, id, Vspec, theta_spec):
        super().__init__(id)
        self.V_spec = Vspec
        self.theta_spec = theta_spec


class Line:
    def __init__(
            self, 
            bus1: Bus, 
            bus2: Bus, 
            r: float, 
            x: float
    ):
        self.bus1 = bus1
        self.bus2 = bus2
        self.r = r                      # line resistance
        self.x = x                      # line reactance

        self.z = self.r + 1j*self.x     # line impedence
        self.y = 1/self.z               # line admittance

# class Load:
#     # Loads will be placed at a bus
#     def __init__(
#             self,
#             associated_bus: Bus,
#             P: float,
#             Q: float
#     ):
#         self.bus = associated_bus
#         self.P = P
#         self.Q = Q


class PowerflowProblem():
    def __init__(
            self,
            lines = List[Line],
            buses = List[Bus],
    ):
        self.lines = lines
        self.buses = buses

        self.PVBuses = []
        self.PQBuses = []
        self.slack_bus = None

        # Populate lists
        for bus in self.buses:
            if isinstance(bus, PVBus):
                self.PVBuses.append(bus)
            elif isinstance(bus, PQBus):
                self.PQBuses.append(bus)
            elif isinstance(bus, SlackBus):     
                
                if not self.slack_bus is None:
                    raise Exception("Multiple slack buses passed")
                self.slack_bus = bus

        # index of each bus in self.buses will be the index in the Y matrix
        # self.bus_id_to_index converts from an id to the index in the Y matrix
        self.bus_id_to_index = {bus.id:i for i, bus in enumerate(self.buses)}
        self.Y = None

        # Check that everything is good
        self.check_valid_problem()

    def check_valid_problem(self) -> None:
        # Make sure that each line connects to nodes that were passed
        bus_ids = set(self.bus_id_to_index.keys())
        for line in self.lines:
            if not line.bus1.id in bus_ids or not line.bus2.id in bus_ids:
                raise Exception("Line element connects to unpassed bus id")

        if self.slack_bus is None:
            raise Exception("No slack bus defined")

    def make_Y_matrix(self) -> np.ndarray:        
        # initialize setup
        N = len(self.buses)
        Y = np.zeros((N, N), dtype=complex)
        for line in self.lines:
            # get index of each bus in the Y matrix
            i = self.bus_id_to_index[line.bus1.id]
            j = self.bus_id_to_index[line.bus2.id]
            # assemble the Y matrix
            Y[i, i] += line.y
            Y[j, j] += line.y
            Y[i, j] -= line.y
            Y[j, i] -= line.y 
        self.Y = Y
        return self.Y