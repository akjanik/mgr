import numpy as np
import matplotlib.pyplot as plt


# Constants
A = 202
B = 1.9
C = 2.04
Q = 343
alpha_w = 0.32
alpha_s = 0.62
R = 800 #??? Znalezione w Internecie dla skaly

# about to change but for now const
eta = 0.25

def f(y, eta):
    """
    y - vector over which we calculate
    returns an array concatenated of two array for y less and greater than
    given eta
    """
    def s(x):
        return 1.241 - 0.723*x**2
    base = Q*s(y)

    less_eta = np.extract(y < eta, y)
    greater_eta = np.extract(y > eta, y)
    return np.concatenate(
        (Q*s(less_eta)*(1 - alpha_w), Q*s(greater_eta)*(1 - alpha_s))
        )

# vector from 0 to 1 with stepsize dy
# decreasing dy sharpens "jump" on the plot
dy = 0.001
y_0_1 = np.linspace(0, 1, int(1/dy))
size_y = len(y_0_1)
pi = np.pi

f_eta = f(y_0_1, eta)

s = len(f_eta)
# case 1: initial condition constant
# T(t,y) = phi(y) = 10

T0 = np.full((1, size_y), 10)

# HERE IS WHAT MAKES PLOTS VERY DIFFERENT
h = 0.1
size_t = 1000
# T = np.zeros((1000,s))
T_current = T0
i = 0
while i < size_t:
    # T[i+1] = T[i] + h/R * (f_eta  - (A + B*T[i]) - C*(T[i] - T0) ) < - old code with big matrix
    T_next = T_current + h/R * (f_eta  - (A + B*T_current) - C*(T_current - T0) )
    T_current = T_next

    i += 1

# TODO: investigate why need to reshape this stupid array
T_next.shape = (int(1/dy),)

plt.plot(y_0_1, T_next, label=r'$T(t,y) = 10$')


# Case2: T(0,y) = sin(pi*y)
T0 = np.zeros((size_y))
# !!!!! Important m parameter
m = len(y_0_1)
k =  1/size_y
for j in range(m):
    # j*k = y_j
    T0[j] = np.sin(pi*j*k)

T_current = T0
i = 0
while i < size_t:
    T_next = T_current + h/R * (f_eta  - (A + B*T_current) - C*(T_current - T0) )
    T_current = T_next
    i += 1
plt.plot(y_0_1, T_next, label=r'$T(t,y) = \sin(\pi y)$')

# Case3: T(0,y) = log(y)
# T0 = np.zeros((size_y))
# for j in range(m):
#     # j*k = y_j
#     T0[j] = np.log(j*k)
#
# T[0] = T0
# for i in range(0, len(T)-1):
#     T[i+1] = T_current + h/R * (f_eta  - (A + B*T_current) - C*(T_current - T0) )
# plt.plot(y_0_1, T[999])
# it works, but with divide by zero error

# Case4: T(t, y) = 0.1*t + y
def calculate_t(t):
    # t here is T_current for each step
    T_t = np.zeros((size_y))
    t = t.transpose()
    # m - length of row vector (number of y division)
    # j/m - from 0 to 1
    for j in range(m):
        T_t[j] = 1/m * t[j]
    return T_t

i = 0
T0 = np.full((1, size_y), 10)
T_current = T0

while i < size_t:
    # dash - kreska
    T_dash = calculate_t(T_current)
    T_next = T_current + h/R * (f_eta  - (A + B*T_current) - C*(T_current - T_dash) )
    T_current = T_next
    i += 1
plt.plot(y_0_1, T_next.transpose(), label='T(t,y) = not sure what')

plt.legend()
plt.text(0.6, 10, r'$\eta = %s$' % eta, fontsize=10)
plt.text(0.6, 8, r'$h = %s, size_t = %s$' %(h, size_t), fontsize=10)
plt.xlabel('y')
plt.ylabel('T')

plt.show()
input()
