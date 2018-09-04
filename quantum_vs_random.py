import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


def disc_line_random_walk(pos, size, steps):
	for _ in range(steps):
		pos += (np.random.rand() >= 0.5)
	return pos % size



# def q_walk_uni_line(state, size, steps):
# 	H = np.array([[1,1], [1, -1]]) / np.sqrt(2)
# 	plus = H * np.array([1,0])

# 	states = np.append(np.array()
# 	walker = np.kron(state, plus)



r_dist = np.array([disc_line_random_walk(51, 101, 40) for i in range(1000)])

fig,axis = plt.subplots()
sbn.distplot(r_dist, ax=axis, kde=True, hist=True)
plt.title("Distribuição empírica para passeio aleatório.")
plt.xlabel("Nó de parada")
plt.ylabel("Frequência relativa")
plt.grid(True) # coller
plt.savefig("rd_walk.png")
plt.clf()

sbn.hist(r_dist)