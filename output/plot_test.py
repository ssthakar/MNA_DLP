import matplotlib.pyplot as plt
import numpy as np

P1 = np.loadtxt("./Pin_1.dat")
P1 = P1 * 0.00075
# P1_avg = np.mean(P1)
# P1_avg = np.full_like(P1, P1_avg)
P2 = np.loadtxt("./Pin_2.dat")
P2 = P2 * 0.00075
# P2_avg = np.mean(P2)
# P2_avg = np.full_like(P2, P2_avg)
# print("Pin_1: ", P1_avg)
# print("Pin_2: ", P2_avg)
plt.plot(P1[:, 0:1], label="Ground truth")
# plt.plot(P1_avg, label="Pin_1_avg")
# plt.plot(P2_avg, label="Pin_2_avg")
plt.plot(P2[:, 0:1], label="unoptimized params")
plt.legend()
plt.grid()
plt.show()
