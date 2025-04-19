"""
powerflow.py

Started by Owen Frausto on April 19, 2025. 
Meant to contain classes for a larger power flow simulator built on top of numpy.

"""

from beartype import beartype, typing
from typing import Literal, List
import numpy as np

# Define bus types 
BusType = Literal["PQ", "PV", "Slack"]

class Bus:
    @beartype
    def __init__(self, bus_id: int, bus_type: BusType):
        self.id = bus_id
        self.type = bus_type


class Line:
    @beartype
    def __init__(
            self, 
            bus1: Bus, 
            bus2: Bus, 
            r: float, 
            x: float
    ):
        self.bus1 = bus1
        self.bus2 = bus2
        self.r = r
        self.x = x


class Load:
    @beartype
    # Loads will be placed at a bus
    def __init__(
            self,
            associated_bus: Bus,
            P: float,
            Q: float
    ):
        self.bus = associated_bus
        self.P = P
        self.Q = Q


class PowerflowProblem():
    @beartype
    def __init__(
            self,
            lines = List[Line],
            buses = List[Bus],
            loads = List[Load]
    ):
        self.lines = lines
        self.buses = buses
        self.loads = loads
        self.Y = None

    def check_valid_problem(self) -> bool:
        for line in self.lines:
            if not line.bus1 in self.buses or not line.bus2 in self.buses:
                return False
        return True

    def make_Y_matrix(self) -> np.ndarray:
        if not self.check_valid_problem():
            raise Exception("Invalid PowerflowProblem")
        
        Y = np.zeros(4)
        raise Exception("Unimplemented: PowerlowProblem make_Y_matrix()")
        self.Y = Y