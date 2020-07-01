from qiskit import *
from qiskit.circuit import *
from qiskit.extensions import RYGate
import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit.providers.aer import *
from qiskit.quantum_info import *
from scipy.optimize import curve_fit

qiskit.__qiskit_version__

# Using Kitaev-Webb algorithm
# Defining functions for mapping qubits and initializing circuits

def g_(sigma_, mu_, lim):
    # normalization function, equal to psi tilde squared, summed over all the integers.
    # in lieu of an infinite sum, we can simply make ``lim" sufficiently high. 
    return np.sum(np.exp((-(np.arange(-lim, lim+1, 1) - mu_)**2)/float((2)*sigma_**2)))

def angle_(sigma_, mu_, lim=10**3):
    # Calculates the angle, based on the square root of probability. 
    # cutoff the infinite sum in g_(...) at 10**3 by default
    return np.arccos(np.sqrt(g_(sigma_/2., mu_/2., lim)/g_(sigma_, mu_, lim)))

def ctrl_states(n):
    states = []
    for i in range(0,2**n):
        s = bin(i).replace("0b", "")
        states.append("".join(["0" for j in range(0,n-len(s))])+s)
    return states

ctrl_states(3)

def new_mu(qub, mu):
    # calculate modified \mu for n-bit string qubit
    # i.e., we have g(b, mu) = (mu - b)/2, b \in {0,1}, and g('',mu) = mu
    # e.g., h('101101', mu) = g('1',g('0',g('1',g(...)))) etc.
    new_mean = mu
    for bit in qub[::-1]:#reversed because we consider the qubits increasingly further back
        new_mean = (new_mean/2.) - ((1/2.)*int(bit))
    return new_mean

def create_circ(N, mu_, sig_,):
    qr = QuantumRegister(N, 'q')
    qc = QuantumCircuit(qr)# Generate a quantum circuit with a quantum register (list) of qubit objects
    alpha_0 = angle_(sig_, mu_) # We multiply by 2, because the ry gate rotates by alpha/2
    qc.ry(2*alpha_0,0) # apply a rotation angle of alpha_0 (multiply by 2 because gate halves parameter)
    for i in range(1,N): # Steps to be done at level q_i
        qstring = ctrl_states(i) # create list of 2^i strings of length i
        for k in qstring:
            alpha_ = angle_(sig_/(2**i), new_mu(k, mu_)) # Calculate angle using modified mean
            new_gate = RYGate(2*alpha_).control(num_ctrl_qubits = i, 
                                                label = None, 
                                                ctrl_state=k) # control state is 
            qc.append(new_gate, qr[:i+1]) # add ry gate to level

    return qc

############################################################################################

N = int(input("Enter number of qubits: "))
sigma = 2**N/6.5
mu = 2**(N-1)
ctrls = ctrl_states(N)

qc = create_circ(N,mu,sigma)


backend = Aer.get_backend('statevector_simulator')
state = execute(qc, backend).result().get_statevector()

print(state)

probs = [x*x for x in state]

print('sum: ' + str(sum(probs)))

# Creating gaussian fitting

def gaussian(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def fit_gaussian(counts):
    x = np.arange(len(counts))
    return curve_fit(gaussian,x,counts)

popt, pcov = fit_gaussian(probs)

print('Amplitude Variance',pcov[0,0])
print('Center  Variance',pcov[1,1])
print('Width Variance',pcov[2,2])

x = np.arange(len(ctrls))
y = gaussian(x, *popt)

# plot the probabilities

# print(np.arange(len(ctrls)))
# print(probs)

plt.figure(figsize=(12, 10))
plt.bar(np.arange(len(ctrls)), probs)
plt.plot(x, y, 'k--')
plt.xticks(np.arange(len(ctrls)), ctrls, rotation=75)
plt.xlabel('List of N-qubit States')
plt.ylabel('Probabilities')
plt.title(str(N) + " Qubits")
plt.show()
