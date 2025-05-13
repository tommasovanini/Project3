import numpy as np

def computeCurrents(t, y, Data):
    # Variables order:
    # VS, VD, w, n, h, c, q, [Ca]
    y0, y1, y2, y3, y4, y5, y6, y7 = y

    gc    = Data['gc'   ]
    gNa   = Data['gNa'  ]
    gK    = Data['gK'   ]
    gSL   = Data['gSL'  ]
    gCa   = Data['gCa'  ]
    gKC   = Data['gKC'  ]
    gKAHP = Data['gKAHP']
    gDL   = Data['gDL'  ]

    ENa = Data['ENa']
    EK  = Data['EK' ]
    ESL = Data['ESL']
    ECa = Data['ECa']
    EDL = Data['EDL']
    
    betam  = Data['betam' ]
    gammam = Data['gammam']

    minf = 0.5*(1 + np.tanh((y0 - betam)/gammam))
    csiCa = min(y7/250, 1)


    IS    = Data['IS']
    ID    = Data['ID']
    IDS   = gc    * (y1-y0)
    INa   = gNa   * minf * (y0-ENa)
    IK    = gK    * y2 * (y0-EK)
    ISL   = gSL   * (y0-ESL)
    ICa   = gCa   * y3 * y4 * (y1-ECa)
    IKC   = gKC   * y5 * csiCa * (y1-EK)
    IKAHP = gKAHP * y6 * (y1-EK)
    IDL   = gDL   * (y1-EDL)

    return IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL


def dydt(t, y, Data):
    y0, y1, y2, y3, y4, y5, y6, y7 = y

    Cm = Data['Cm']
    p  = Data['p']

    phiw   = Data['phiw'  ]
    betaw  = Data['betaw' ]
    gammaw = Data['gammaw']
    tauw   = 1/np.cosh((y0 - betaw)/gammaw*0.5)

    taun = Data['taun']
    tauh = Data['tauh']

    winf = 0.5*(1 + np.tanh((y0 - betaw)/gammaw))
    # To avoid overflow in exp
    if(y1 >= 0):
        hinf = 0
        ninf = 1/(1 + np.exp(-(y1 +  9)/0.5))
    elif(y1 <= -20):
        hinf = 1/(1 + np.exp( (y1 + 21)/0.5))
        ninf = 0
    else:
        hinf = 1/(1 + np.exp( (y1 + 21)/0.5))
        ninf = 1/(1 + np.exp(-(y1 +  9)/0.5))

    if y1 <= 50:
        alphac = (np.exp((y1 - 10)/11 - (y1 - 6.5)/27))/18.975
        betac = 2*np.exp((6.5 - y1)/27) - alphac
    else:
        alphac = 2*np.exp((6.5 - y1)/27)
        betac = 0

    alphaq = min(0.00002*y7, 0.01)
    betaq  = 0.001

    IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL = computeCurrents(t, y, Data)

    # Print the currents every 10 timesteps, up to 10ms
    if t <= 10 and (int(100*t))%10 == 0:
        print("{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL))

    # Variables order: VS, VD, w, n, h, c, q, [Ca]
    dy0 = 1.0/Cm * (IS/p     + IDS/p     - INa - IK  - ISL)
    dy1 = 1.0/Cm * (ID/(1-p) - IDS/(1-p) - ICa - IKC - IKAHP - IDL)
    dy2 = phiw * (winf-y2)/tauw
    dy3 = (ninf-y3)/taun
    dy4 = (hinf-y4)/tauh
    dy5 = alphac*(1-y5) - betac*y5
    dy6 = alphaq*(1-y6) - betaq*y6
    dy7 = -0.13*ICa - 0.075*y7

    # print(IS/p, IDS/p, -INa, -IK, -ISL)
    # print(ID/(1-p), IDS/(1-p), -ICa, -IKC, -IKAHP, -IDL)

    # print("dy0:", dy0, "dy1:", dy1, "dy2:", dy2, "\ndy3:", dy3, "dy4:", dy4, "dy5:", dy5, "\ndy6:", dy6, "dy7:", dy7)
    # print("\n")

    return [dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7]