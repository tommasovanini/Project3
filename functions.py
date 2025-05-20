import numpy as np

def computeCurrents(t, y, Data):
    # t is a float
    # y.shape == (8,)
    # Data is a dictionary with all the parameters

    VS, VD, w, n, h, c, q, Ca = y

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

    minf = 0.5*(1 + np.tanh((VS - betam)/gammam))
    csiCa = (0.004*Ca + 1 - np.abs(0.004*Ca - 1))*0.5       # equivalent to min(Ca/250, 1)


    IS    = Data['IS']
    ID    = Data['ID']
    IDS   = gc    * (VD-VS)
    INa   = gNa   * minf * (VS-ENa)
    IK    = gK    * w * (VS-EK)
    ISL   = gSL   * (VS-ESL)
    ICa   = gCa   * n * h * (VD-ECa)
    IKC   = gKC   * c * csiCa * (VD-EK)
    IKAHP = gKAHP * q * (VD-EK)
    IDL   = gDL   * (VD-EDL)

    return IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL


def computeCurrents_vectorized(t, y, Data):
    # t.shape == (n,)
    # y.shape == (8,n)
    # Data is a dictionary with all the parameters

    IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL = np.array([computeCurrents(t[i], y[:, i], Data) for i in range(t.shape[0])]).T

    return IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL


def dydt(t, y, Data, return_full=False):
    # t is a float
    # y.shape == (8,)
    # Data is a dictionary with all the parameters
    
    VS, VD, w, n, h, c, q, Ca = y

    Cm = Data['Cm']
    p  = Data['p']

    phiw   = Data['phiw'  ]
    betaw  = Data['betaw' ]
    gammaw = Data['gammaw']
    tauw   = 1/np.cosh((VS - betaw)/gammaw*0.5)

    taun = Data['taun']
    tauh = Data['tauh']

    winf = 0.5*(1 + np.tanh((VS - betaw)/gammaw))
    hinf = 1/(1 + np.exp( (VD + 21)*2))
    ninf = 1/(1 + np.exp(-(VD +  9)*2))

    if VD <= 50:
        alphac = (np.exp((VD - 10)/11 - (VD - 6.5)/27))/18.975
        betac  = 2*np.exp((6.5 - VD)/27) - alphac
    else:
        alphac = 2*np.exp((6.5 - VD)/27)
        betac  = 0

    alphaq = (0.00002*Ca + 0.01 - np.abs(0.00002*Ca - 0.01))*0.5      # equivalent to min(0.00002*Ca, 0.01)
    betaq  = 0.001

    IS, ID, IDS, INa, IK, ISL, ICa, IKC, IKAHP, IDL = computeCurrents(t, y, Data)

    dVSdt = 1.0/Cm * (IS/p     + IDS/p     - INa - IK  - ISL)
    dVDdt = 1.0/Cm * (ID/(1-p) - IDS/(1-p) - ICa - IKC - IKAHP - IDL)
    dwdt  = phiw * (winf-w)/tauw
    dndt  = (ninf-n)/taun
    dhdt  = (hinf-h)/tauh
    dcdt  = alphac*(1-c) - betac*c
    dqdt  = alphaq*(1-q) - betaq*q
    dCadt = -0.13*ICa - 0.075*Ca

    if not return_full:
        return [dVSdt, dVDdt, dwdt, dndt, dhdt, dcdt, dqdt, dCadt]
    else:
        pieces = []
        pieces.append([1.0/Cm*IS/p, 1.0/Cm*IDS/p, -1.0/Cm*INa, -1.0/Cm*IK, -1.0/Cm*ISL])
        pieces.append([1.0/Cm*ID/(1-p), -1.0/Cm*IDS/(1-p), -1.0/Cm*ICa, -1.0/Cm*IKC, -1.0/Cm*IKAHP, -1.0/Cm*IDL])
        pieces.append([winf, tauw])
        pieces.append([(ninf-n)/taun])
        pieces.append([(hinf-h)/tauh])
        pieces.append([alphac, betac])
        pieces.append([alphaq, betaq])
        pieces.append([-0.13*ICa, -0.075*Ca])
        return [dVSdt, dVDdt, dwdt, dndt, dhdt, dcdt, dqdt, dCadt, pieces]


def dydt_vectorized(t, y, Data, return_full=False):
    # t.shape == (n,)
    # y.shape == (8,n)
    # Data is a dictionary with all the parameters

    if not return_full:
        dVSdt, dVDdt, dwdt, dndt, dhdt, dcdt, dqdt, dCadt = \
            np.array([dydt(t[i], y[:, i], Data) for i in range(t.shape[0])]).T
        return dVSdt, dVDdt, dwdt, dndt, dhdt, dcdt, dqdt, dCadt

    else:
        dVSdt, dVDdt, dwdt, dndt, dhdt, dcdt, dqdt, dCadt = np.zeros((8, t.shape[0]))
        pieces = []
        pieces.append(np.zeros((5, t.shape[0])))
        pieces.append(np.zeros((6, t.shape[0])))
        pieces.append(np.zeros((2, t.shape[0])))
        pieces.append(np.zeros((1, t.shape[0])))
        pieces.append(np.zeros((1, t.shape[0])))
        pieces.append(np.zeros((2, t.shape[0])))
        pieces.append(np.zeros((2, t.shape[0])))
        pieces.append(np.zeros((2, t.shape[0])))

        for i in range(t.shape[0]):
            dVSdt[i], dVDdt[i], dwdt[i], dndt[i], dhdt[i], dcdt[i], dqdt[i], dCadt[i], pieces_i = \
                dydt(t[i], y[:, i], Data, return_full=True)
            for j in range(len(pieces)):
                pieces[j][:, i] = pieces_i[j]

        return dVSdt, dVDdt, dwdt, dndt, dhdt, dcdt, dqdt, dCadt, pieces
