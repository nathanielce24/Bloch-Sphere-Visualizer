import numpy as np
from qc_logic import *

class SystemState:
    
    def __init__(self):
        self.initial_state = self.ket_0
        self.states = [self.ket_0]
        self.gates = []
        self.initial_transformation_decomposition = decompose_matrix(np.eye(2))
        self.final_matrix = np.array(np.eye(2))
        self.theta = 0
        self.phi = 0

    def change_initial_state(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.initial_state = get_quantum_state(theta,phi)
        self.initial_transformation_decomposition = decompose_matrix(self.get_initial_matrix())
        self.update_states()

    def get_initial_matrix(self):
        return get_transformation_matrix(self.theta, self.phi)
    
    def get_final_matrix(self):
        curr = self.get_initial_matrix()
        for gate in self.gates:
            curr = gate @ curr
        return curr

    def update_states(self):
        curr = self.initial_state
        new_states = [curr]
        for gate in self.gates:
            next = gate @ curr
            new_states.append(next)
            curr = next
        self.states = new_states
        self.final_matrix = self.get_final_matrix()
        return self.states


    def add_gate(self, gate):
        self.gates.append(gate)
        self.states.append(gate @ self.states[-1])
        self.final_matrix = gate @ self.final_matrix

    def remove_gate(self, index):
        if -len(self.gates) <= index < len(self.gates):
            self.gates.pop(index)
            self.update_states()



 



    
