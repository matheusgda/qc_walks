import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from projectq import MainEngine
from projectq.ops import Measure, X

from ring_quantum_walk import walk


def disc_line_random_walk(pos, size, steps):
	for _ in range(steps):
		pos += (np.random.rand() >= 0.5)
	return pos % size



def prepare_state(state, eng):
    n_qbits = len(state)
    reg = eng.allocate_qureg(n_qbits)
    for i in range(n_qbits):
        if state[i]:
            X | reg[i]
    return reg




# def q_walk_uni_line(state, size, steps):
# 	H = np.array([[1,1], [1, -1]]) / np.sqrt(2)
# 	plus = H * np.array([1,0])

# 	states = np.append(np.array()
# 	walker = np.kron(state, plus)


samples = 100

r_dist = np.array([disc_line_random_walk(51, 101, 40) for i in range(samples)])

fig,axis = plt.subplots()
sbn.distplot(r_dist, ax=axis, kde=True, hist=True)
plt.title(
    "Distribuição empírica para passeio aleatório com {} amostras."
    .format(samples))
plt.xlabel("Nó de parada")
plt.ylabel("Frequência relativa")
plt.grid(True) # coller
plt.savefig("rd_walk{}.png".format(samples))
plt.clf()



n_qbits = 8
graph_size = 101
steps = 40


eng = MainEngine()
data = np.zeros(samples, dtype=int)
init_mask = np.array([0, 1, 1, 0, 0, 1, 1, 0])

for i in range(samples):
    nodes = prepare_state(init_mask, eng)
    walk(eng, nodes, n_qbits, steps, graph_size)
    Measure | (nodes)
    eng.flush()
    last_st = 0
    for j in range(len(nodes) - 1):
        last_st += int(nodes[j]) * 2 ** (n_qbits - 2 - j)
    print(last_st, "state found")
    data[i] = last_st


print(data)
fig,axis = plt.subplots()
sbn.distplot(data, ax=axis, kde=True, hist=True)
plt.title(
    "Distribuição empírica para passeio quântico com {} amostras."
    .format(samples))
plt.xlabel("Nó de parada")
plt.ylabel("Frequência relativa")
plt.grid(True) # coller
plt.savefig("quantum_walk.png")
plt.clf()

        