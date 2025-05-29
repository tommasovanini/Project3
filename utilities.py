import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def EI(t, f, y0, Data):

    y = np.zeros((len(y0), len(t)))
    y[:, 0] = y0
    dt = t[1] - t[0]

    delay = Data['delay']

    for n in range(len(t) - 1):
        t_np1 = t[n+1]
        y_n = y[:, n]

        def res(y_np1):
        
            f_np1 = np.array(f(t_np1, y_np1, Data)).T

            # This is needed to be able to use either EI or solve_ivp in the main
            f_np1[8] -= -1.0/Data['Cm'] * Data['g_12'] * (y_np1[1] - Data['V_12'])
            f_np1[0] -= -1.0/Data['Cm'] * Data['g_21'] * (y_np1[9] - Data['V_21'])

            if t_np1 > delay:
                f_np1[8] += -1.0/Data['Cm'] * Data['g_12'] * (y[1, n+1-int(dt*delay)] - Data['V_12'])
                # f_np1[0] += -1.0/Data['Cm'] * Data['g_21'] * (y[9, n+1-int(dt*delay)] - Data['V_21'])

            return y_np1 - (y_n + dt * f_np1)
        
        y[:, n+1] = fsolve(func=res,
                           x0=y_n)
    
    class Solution:
        def __init__(self, success, message, y, t):
            self.success = success
            self.message = message
            self.y = y
            self.t = t

    return Solution(success='True', message='The solver converged.', y=y, t=t)


def plot_solutions(all_data, g_values, colors):
    
    time = all_data['time']

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VS_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{S, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 2)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VD_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{D, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 3)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['ICa_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$I_{Ca, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 4)
    for i, g_val in enumerate(g_values):
        plt.plot(time, -all_data['IDS_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$-I_{DS, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VS_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{S, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 2)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VD_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{D, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 3)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['ICa_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$I_{Ca, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 4)
    for i, g_val in enumerate(g_values):
        plt.plot(time, -all_data['IDS_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$-I_{DS, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.show()



    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, g_val in enumerate(g_values):
        ax.plot(time, all_data['VS_1'][i], all_data['w_1'][i], color=colors[i], label=f'g={g_val}')
    ax.view_init(10, 260)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('$V_{S, 1}$ (mV)')
    ax.set_zlabel('$w_1$')
    plt.title("t vs $V_{S, 1}$ vs $w_1$")
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.grid()
    plt.show()



    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, g_val in enumerate(g_values):
        ax.plot(time, all_data['VS_2'][i], all_data['w_2'][i], color=colors[i], label=f'g={g_val}')
    ax.view_init(10, 260)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('$V_{S, 2}$ (mV)')
    ax.set_zlabel('$w_2$')
    plt.title("t vs $V_{S, 2}$ vs $w_2$")
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.grid()
    plt.show()