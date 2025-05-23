import numpy as np
from scipy.optimize import fsolve

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


            ret = y_np1 - (y_n + dt * f_np1)
            return ret
        

        y[:, n+1] = fsolve(func=res,
                           x0=y_n)
    
    class Solution:
        def __init__(self, success, message, y, t):
            self.success = success
            self.message = message
            self.y = y
            self.t = t

    return Solution(success='True', message='The solver converged.', y=y, t=t)