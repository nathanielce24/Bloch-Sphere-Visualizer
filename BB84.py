import numpy as np
import random
from qc_logic import get_quantum_state, ket_0, get_transformation_matrix, get_probs, ket_minus, ket_plus, ket_1, H, measure
from visualization import draw_state_arc, plot_sphere, plot_state

def get_bit_string(num):
 bits = []
 for i in range(num):
    bits.append(random.randint(0,1))
 return bits


def get_bases_set(num):
  bases = []
  for i in range(num):
    bases.append(random.choice(["R","D"]))
  return bases

def encode(bitstring):
  bases = get_bases_set(len(bitstring))
  encoded = []
  for i in range(len(bitstring)):
    if bitstring[i] == 1:
      if bases[i]=="R":
        encoded.append(ket_1)
      else:
        encoded.append(ket_minus)
    else:
      if bases[i]=="R":
        encoded.append(ket_0)
      else:
        encoded.append(ket_plus)
  return bases, encoded

def decode_r(state):
  return np.eye(2) @ state

def decode_d(state):
  return H @ state

def random_decode(states):
  bases = get_bases_set(len(states))
  measurements = []
  for i in range(len(states)):
    if bases[i]=="R":
      decoded_state = decode_r(states[i])
    else:
      decoded_state = decode_d(states[i])
    measurements.append(measure(decoded_state))
  return bases, measurements

def compare_bases(b1, b2):
  res = []
  for i in range(len(b1)):
    res.append(b1[i] == b2[i])
  return res

def get_final_bits(measurements, correct_bases):
  res = []
  for i in range(len(measurements)):
    if correct_bases[i]:
      res.append(measurements[i])
  return res


n = 20
alice_bits = get_bit_string(n)
alice_bases, encoded_states = encode(alice_bits)
bob_bases, bob_measurements = random_decode(encoded_states)

matches = compare_bases(alice_bases, bob_bases)
alice_sifted = get_final_bits(alice_bits, matches)
bob_sifted   = get_final_bits(bob_measurements, matches)

agree = sum(a == b for a, b in zip(alice_sifted, bob_sifted))
match_rate = agree / len(alice_sifted) if alice_sifted else float('nan')

print(f"Alice bits:      {alice_bits}")
print(f"Alice bases:     {alice_bases}")
print(f"Bob bases:       {bob_bases}")
print(f"Bob measurements:{bob_measurements}")
print(f"Basis matches:   {matches}")
print(f"Alice sifted:    {alice_sifted}")
print(f"Bob sifted:      {bob_sifted}")
print(f"Sifted key length: {len(alice_sifted)} / {n}")
print(f"Bit match rate:  {match_rate:.0%}")