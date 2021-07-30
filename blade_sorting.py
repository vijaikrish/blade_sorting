import numpy as np
import random
from matplotlib import pyplot as plt

# Blade weight list
blade_weight = [1762.04, 1761.98, 1726.72, 1755.64, 1755, 1756.88, 1762.72, 1762.2, 1762.72, 1758.24, 1758.04, 1740.86,
                1754.68, 1742.08, 1754.58, 1754.38, 1757.76, 1762.28, 1769.66, 1754.92, 1757.9, 1761.32, 1755.7, 1754.7,
                1749.6, 1779.36, 1756.4, 1767.28, 1756.54, 1758.02, 1772.24, 1772.96]  # Unit in gram-mm

# Radius of the rotor disk
r = 1  # Unit in mm

# using random.sample() to shuffle the blade weight
ran_old = blade_weight
# Iterative loop is used for the shuffling and the values are updated in order to avoid the repeat the values
for i in range(0, 1000):
    ran_new = random.sample(ran_old, len(blade_weight))
    ran_old = ran_new
    # print(ran_new)
    ran_cal = np.array(ran_old)
    n = len(ran_cal)
    m_x = np.zeros(n)
    m_y = np.zeros(n)
    # Calculate the unbalance of the total system in iterative loop
    for j in range(0, n):
        m_x[j] = ran_new[j] * r * np.cos((2 * 180 * j) / n)
        m_y[j] = ran_new[j] * r * np.sin((2 * 180 * j) / n)
        M_x = np.sum(m_x)
        M_y = np.sum(m_y)
        res_un = np.sqrt(M_x ** 2 + M_y ** 2)

    # print(i)  # Unit in g-mm
    if i == 0:
        res_op = res_un

    if res_un < res_op:
        res_op = res_un
        print(1 / res_op)
        # res_op = np.append(res_un)
        blade_op = ran_new
        m_x_op = m_x
        m_y_op = m_y
        M_x_op = np.sum(m_x_op)
        M_y_op = np.sum(m_y_op)
        x = (M_y_op / M_x_op)
        angle = (180 / np.pi) * np.arctan(x)

print(blade_op, res_op, angle)

plt.figure()
ay = plt.subplot(111, projection='polar')
for k in range(0, n):
    ay.plot([0, (k / n) * (2 * 3.14)], [0, blade_op[k]], 'b-', lw=2, label="J")

ay.plot([0, angle * (2 * (3.14 / 360))], [0, res_op / r], 'r-', lw=2, label="J")

angle = np.deg2rad(67.5)
ay.legend(loc="lower left", bbox_to_anchor=(.5 + np.cos(angle) / 2, .5 + np.sin(angle) / 2))
# ay.set_thetagrids(-10)
# ay.set_rlabel_position(-22.5)
# thetaticks = np.arange(0, 360, 45)
# ay.set_thetagrids(thetaticks, labels=None)
plt.title("POLAR PLOT")
# plt.grid(axis='x')

plt.show()
