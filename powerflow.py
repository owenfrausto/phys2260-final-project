"""
powerflow.py

Started by Owen Frausto on April 19, 2025. 
Meant to contain classes for a larger power flow simulator built on top of numpy.

"""

from beartype import beartype, typing
from typing import Literal, List
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

# Bus parent class
class Bus:
    def __init__(self, bus_id: int) -> None:
        self.id = bus_id

# Define bus subclasses
class PQBus(Bus):
    def __init__(self, id, Pd, Qd) -> None:
        super().__init__(id)
        self.Pd = Pd
        self.Qd = Qd

class PVBus(Bus):
    def __init__(self, id, Pg, Vspec) -> None:
        super().__init__(id)
        self.Pg = Pg
        self.Vspec = Vspec

class SlackBus(Bus):
    def __init__(self, id, Vspec, theta_spec) -> None:
        super().__init__(id)
        self.Vspec = Vspec
        self.theta_spec = theta_spec


class Line:
    def __init__(
            self, 
            bus1: Bus, 
            bus2: Bus, 
            r: float, 
            x: float
    ) -> None:
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
    ) -> None:
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

    def solve_system(self, tol: float = 1e-6, max_iter: int = 20):
        Y = self.make_Y_matrix()
        N = len(self.buses)

        # Initial flat start
        V = np.ones(N, dtype=complex)
        slack_idx = self.bus_id_to_index[self.slack_bus.id]
        V[slack_idx] = self.slack_bus.Vspec * np.exp(1j * self.slack_bus.theta_spec)

        pv_indices = [self.bus_id_to_index[bus.id] for bus in self.PVBuses]
        pq_indices = [self.bus_id_to_index[bus.id] for bus in self.PQBuses]

        for iteration in range(max_iter):
            # Calculate power injections
            I = Y @ V
            S = V * np.conj(I)
            P = S.real
            Q = S.imag

            # Build mismatch vector
            mismatch = []
            for idx in pv_indices + pq_indices:
                bus = self.buses[idx]
                Pd = getattr(bus, "Pd", 0.0)
                Pg = getattr(bus, "Pg", 0.0)
                mismatch.append(Pg - Pd - P[idx])
            for idx in pq_indices:
                bus = self.buses[idx]
                Qd = getattr(bus, "Qd", 0.0)
                mismatch.append(-Qd - Q[idx])
            mismatch = np.array(mismatch)

            max_mismatch = np.max(np.abs(mismatch))
            print(f"Iteration {iteration}, max mismatch = {max_mismatch:.6e}")
            if max_mismatch < tol:
                break

            # Build Jacobian matrix
            n_pv_pq = len(pv_indices) + len(pq_indices)
            n_pq = len(pq_indices)
            J = np.zeros((n_pv_pq + n_pq, n_pv_pq + n_pq))

            angles = np.angle(V)
            magnitudes = np.abs(V)

            # dP/dTheta
            for a, idx_i in enumerate(pv_indices + pq_indices):
                for b, idx_j in enumerate(pv_indices + pq_indices):
                    if idx_i == idx_j:
                        J[a, b] = -Q[idx_i] - magnitudes[idx_i]**2 * Y[idx_i, idx_i].imag
                    else:
                        J[a, b] = magnitudes[idx_i] * magnitudes[idx_j] * (
                            Y[idx_i, idx_j].real * np.sin(angles[idx_i] - angles[idx_j]) -
                            Y[idx_i, idx_j].imag * np.cos(angles[idx_i] - angles[idx_j])
                        )

            # dP/dV
            for a, idx_i in enumerate(pq_indices):
                for b, idx_j in enumerate(pv_indices + pq_indices):
                    if idx_i == idx_j:
                        J[a + n_pv_pq, b] = P[idx_i] - magnitudes[idx_i]**2 * Y[idx_i, idx_i].real
                    else:
                        J[a + n_pv_pq, b] = -magnitudes[idx_i] * magnitudes[idx_j] * (
                            Y[idx_i, idx_j].real * np.cos(angles[idx_i] - angles[idx_j]) +
                            Y[idx_i, idx_j].imag * np.sin(angles[idx_i] - angles[idx_j])
                        )

            # dQ/dTheta
            for a, idx_i in enumerate(pv_indices + pq_indices):
                for b, idx_j in enumerate(pq_indices):
                    if idx_i == idx_j:
                        J[a, b + n_pv_pq] = P[idx_i] / magnitudes[idx_i] + magnitudes[idx_i] * Y[idx_i, idx_i].real
                    else:
                        J[a, b + n_pv_pq] = magnitudes[idx_i] * (
                            Y[idx_i, idx_j].real * np.cos(angles[idx_i] - angles[idx_j]) +
                            Y[idx_i, idx_j].imag * np.sin(angles[idx_i] - angles[idx_j])
                        )

            # dQ/dV
            for a, idx_i in enumerate(pq_indices):
                for b, idx_j in enumerate(pq_indices):
                    if idx_i == idx_j:
                        J[a + n_pv_pq, b + n_pv_pq] = Q[idx_i] / magnitudes[idx_i] - magnitudes[idx_i] * Y[idx_i, idx_i].imag
                    else:
                        J[a + n_pv_pq, b + n_pv_pq] = magnitudes[idx_i] * (
                            Y[idx_i, idx_j].real * np.sin(angles[idx_i] - angles[idx_j]) -
                            Y[idx_i, idx_j].imag * np.cos(angles[idx_i] - angles[idx_j])
                        )

            # Solve for updates
            delta = np.linalg.solve(J, mismatch)

            d_theta = delta[:n_pv_pq]
            d_V = delta[n_pv_pq:]

            # Update voltage
            for k, idx in enumerate(pv_indices + pq_indices):
                angles[idx] += d_theta[k]
            for k, idx in enumerate(pq_indices):
                magnitudes[idx] += d_V[k]

            V = magnitudes * np.exp(1j * angles)

        self.solution = V

    def visualize(self, title:str) -> None:
        # create a graph with networkx

        G = nx.Graph()

        # add nodes and colors
        color_map = []
        for bus in self.buses:
            G.add_node(bus.id, obj=bus)
            if isinstance(bus, SlackBus):
                color_map.append("orange")
            elif isinstance(bus, PVBus):
                color_map.append("blue")
            else:
                # is PQ bus
                color_map.append("green")

        # add edges
        for line in self.lines:
            G.add_edge(line.bus1.id, line.bus2.id, y=line.y)

        nx.draw(G, with_labels=True, node_color=color_map)
        from matplotlib.patches import Patch

        # add in legend
        import matplotlib.pyplot as plt
        legend_elements = [
            Patch(facecolor="orange", edgecolor='k', label="Slack"),
            Patch(facecolor="blue", edgecolor='k', label="Generator (PV)"),
            Patch(facecolor="green", edgecolor='k', label="Load (PQ)")
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        plt.title(title)
        plt.show()

    def print_solution(self):
        import numpy as np

        if not hasattr(self, "solution"):
            raise Exception("No solution found. Run solve_system() first.")

        V = self.solution
        Y = self.Y
        print(f"{'Bus':>5} {'Type':>10} {'|V| (p.u.)':>12} {'Angle (deg)':>12} {'Power Factor':>14}")

        for bus in self.buses:
            idx = self.bus_id_to_index[bus.id]
            mag = np.abs(V[idx])
            angle_deg = np.angle(V[idx], deg=True)

            if isinstance(bus, SlackBus):
                btype = "Slack"
            elif isinstance(bus, PVBus):
                btype = "Generator"
            else:  # PQBus
                btype = "Load"

            # Calculate S = V * conj(YV) to find real and reactive power
            I = Y[idx, :] @ V
            S = V[idx] * np.conj(I)
            P = S.real
            Q = S.imag

            # Power factor: cos(phi) = P / sqrt(P^2 + Q^2)
            S_magnitude = np.sqrt(P**2 + Q**2)
            if S_magnitude > 1e-8:
                pf = P / S_magnitude
            else:
                pf = 1.0  # no load

            print(f"{bus.id:>5} {btype:>10} {mag:>12.6f} {angle_deg:>12.6f} {pf:>14.6f}")
