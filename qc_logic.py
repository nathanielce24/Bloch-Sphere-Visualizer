import numpy as np
from visualization import convert_to_cartesian

ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

ket_plus = np.array([1,1], dtype=complex)*1/np.sqrt(2)
ket_minus = np.array([1,-1], dtype=complex)*1/np.sqrt(2)

H = np.array([[1,  1],[1, -1]]) / np.sqrt(2)
PX = np.array([[0, 1],[1,0]])
PY = np.array([[0,  -1j],[1j, 0]])
PZ = np.array([[1, 0],[0,-1]])

def get_quantum_state(theta, phi):
 return np.cos(theta/2)*ket_0+np.exp(1j*phi)*np.sin(theta/2)*ket_1

def get_probs(state):
 return np.square(np.abs(state))

def measure(state):
 probs = get_probs(state).astype(float)
 probs = probs / np.sum(probs)
 return int(np.random.choice([0, 1], p=probs))

def state_to_polar(state):
  theta = 2 * np.arccos(np.abs(state[0]))
  phi = np.angle(state[1]) - np.angle(state[0])
  return theta, phi % (2 * np.pi)

def state_to_cartesian(state):
  theta, phi = state_to_polar(state)
  x,y,z = convert_to_cartesian(1, phi, theta)
  return x,y,z

def get_transformation_matrix(theta, phi):
  U = np.array(
            [[np.cos(theta/2),                 -np.exp(-1j*phi)*np.sin(theta/2)],
            [np.exp(-1j*phi)*np.sin(theta/2),                  np.cos(theta/2)]])
  return U

def apply_gate(gate, state):
 return gate @ state

def decompose_matrix(matrix):
 a = matrix[1,1]
 b = matrix [1,2]
 c = matrix [2,1]
 d = matrix [2,2]

 gamma = 2*np.arctan(np.abs(b) / np.abs(a))
 

def add_gate():
 pass
