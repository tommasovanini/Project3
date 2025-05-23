import numpy as np

def computeCurrents(t, y, Data):
    # t is a float
    # y.shape == (16,)
    # Data is a dictionary with all the parameters

    VS_1, VD_1, w_1, n_1, h_1, c_1, q_1, Ca_1 = y[:8]
    VS_2, VD_2, w_2, n_2, h_2, c_2, q_2, Ca_2 = y[8:]

    gc      = Data['gc'   ]
    gNa     = Data['gNa'  ]
    gK      = Data['gK'   ]
    gSL     = Data['gSL'  ]
    gCa_1   = Data['gCa_1']
    gCa_2   = Data['gCa_2']
    gKC     = Data['gKC'  ]
    gKAHP   = Data['gKAHP']
    gDL     = Data['gDL'  ]

    ENa = Data['ENa']
    EK  = Data['EK' ]
    ESL = Data['ESL']
    ECa = Data['ECa']
    EDL = Data['EDL']
    
    betam  = Data['betam' ]
    gammam = Data['gammam']

    minf_1 = 0.5*(1 + np.tanh((VS_1 - betam)/gammam))
    minf_2 = 0.5*(1 + np.tanh((VS_2 - betam)/gammam))
    csiCa_1 = (0.004*Ca_1 + 1 - np.abs(0.004*Ca_1 - 1))*0.5
    csiCa_2 = (0.004*Ca_2 + 1 - np.abs(0.004*Ca_2 - 1))*0.5


    IS_1    = Data['IS_1']
    ID_1    = Data['ID_1']
    IDS_1   = gc    * (VD_1-VS_1)
    INa_1   = gNa   * minf_1 * (VS_1-ENa)
    IK_1    = gK    * w_1 * (VS_1-EK)
    ISL_1   = gSL   * (VS_1-ESL)
    ICa_1   = gCa_1   * n_1 * h_1 * (VD_1-ECa)
    IKC_1   = gKC   * c_1 * csiCa_1 * (VD_1-EK)
    IKAHP_1 = gKAHP * q_1 * (VD_1-EK)
    IDL_1   = gDL   * (VD_1-EDL)

    
    IS_2    = Data['IS_2']
    ID_2    = Data['ID_2']
    IDS_2   = gc    * (VD_2-VS_2)
    INa_2   = gNa   * minf_2 * (VS_2-ENa)
    IK_2    = gK    * w_2 * (VS_2-EK)
    ISL_2   = gSL   * (VS_2-ESL)
    ICa_2   = gCa_2   * n_2 * h_2 * (VD_2-ECa)
    IKC_2   = gKC   * c_2 * csiCa_2 * (VD_2-EK)
    IKAHP_2 = gKAHP * q_2 * (VD_2-EK)
    IDL_2   = gDL   * (VD_2-EDL)

    return  IS_1, ID_1, IDS_1, INa_1, IK_1, ISL_1, ICa_1, IKC_1, IKAHP_1, IDL_1, \
            IS_2, ID_2, IDS_2, INa_2, IK_2, ISL_2, ICa_2, IKC_2, IKAHP_2, IDL_2


def computeCurrents_vectorized(t, y, Data):
    # t.shape == (n,)
    # y.shape == (16,n)
    # Data is a dictionary with all the parameters

    IS_1, ID_1, IDS_1, INa_1, IK_1, ISL_1, ICa_1, IKC_1, IKAHP_1, IDL_1, \
        IS_2, ID_2, IDS_2, INa_2, IK_2, ISL_2, ICa_2, IKC_2, IKAHP_2, IDL_2 = \
            np.array([computeCurrents(t[i], y[:, i], Data) for i in range(t.shape[0])]).T

    return IS_1, ID_1, IDS_1, INa_1, IK_1, ISL_1, ICa_1, IKC_1, IKAHP_1, IDL_1, \
              IS_2, ID_2, IDS_2, INa_2, IK_2, ISL_2, ICa_2, IKC_2, IKAHP_2, IDL_2


def dydt(t, y, Data):
    # t is a float
    # y.shape == (16,)
    # Data is a dictionary with all the parameters
    
    VS_1, VD_1, w_1, n_1, h_1, c_1, q_1, Ca_1 = y[:8]
    VS_2, VD_2, w_2, n_2, h_2, c_2, q_2, Ca_2 = y[8:]

    Cm = Data['Cm']
    p  = Data['p']

    phiw   = Data['phiw'  ]
    betaw  = Data['betaw' ]
    gammaw = Data['gammaw']
    tauw_1   = 1/np.cosh((VS_1 - betaw)/gammaw*0.5)
    tauw_2   = 1/np.cosh((VS_2 - betaw)/gammaw*0.5)

    taun = Data['taun']
    tauh = Data['tauh']

    winf_1 = 0.5*(1 + np.tanh((VS_1 - betaw)/gammaw))
    hinf_1 = 1/(1 + np.exp( (VD_1 + 21)*2))
    ninf_1 = 1/(1 + np.exp(-(VD_1 +  9)*2))
    winf_2 = 0.5*(1 + np.tanh((VS_2 - betaw)/gammaw))
    hinf_2 = 1/(1 + np.exp( (VD_2 + 21)*2))
    ninf_2 = 1/(1 + np.exp(-(VD_2 +  9)*2))

    if VD_1 <= 50:
        alphac_1 = (np.exp((VD_1 - 10)/11 - (VD_1 - 6.5)/27))/18.975
        betac_1  = 2*np.exp((6.5 - VD_1)/27) - alphac_1
    else:
        alphac_1 = 2*np.exp((6.5 - VD_1)/27)
        betac_1  = 0

    if VD_2 <= 50:
        alphac_2 = (np.exp((VD_2 - 10)/11 - (VD_2 - 6.5)/27))/18.975
        betac_2  = 2*np.exp((6.5 - VD_2)/27) - alphac_2
    else:
        alphac_2 = 2*np.exp((6.5 - VD_2)/27)
        betac_2  = 0

    alphaq_1 = (0.00002*Ca_1 + 0.01 - np.abs(0.00002*Ca_1 - 0.01))*0.5
    betaq_1  = 0.001
    alphaq_2 = (0.00002*Ca_2 + 0.01 - np.abs(0.00002*Ca_2 - 0.01))*0.5
    betaq_2  = 0.001

    IS_1, ID_1, IDS_1, INa_1, IK_1, ISL_1, ICa_1, IKC_1, IKAHP_1, IDL_1, \
        IS_2, ID_2, IDS_2, INa_2, IK_2, ISL_2, ICa_2, IKC_2, IKAHP_2, IDL_2 = \
            computeCurrents(t, y, Data)

    I12 = Data['g_12'] * (VD_1-Data['V_12'])
    I21 = Data['g_21'] * (VD_2-Data['V_21'])

    dVSdt_1 = 1.0/Cm * (IS_1/p     + IDS_1/p     - INa_1 - IK_1  - ISL_1   - I21)
    dVDdt_1 = 1.0/Cm * (ID_1/(1-p) - IDS_1/(1-p) - ICa_1 - IKC_1 - IKAHP_1 - IDL_1)
    dwdt_1  = phiw * (winf_1-w_1)/tauw_1
    dndt_1  = (ninf_1-n_1)/taun
    dhdt_1  = (hinf_1-h_1)/tauh
    dcdt_1  = alphac_1*(1-c_1) - betac_1*c_1
    dqdt_1  = alphaq_1*(1-q_1) - betaq_1*q_1
    dCadt_1 = -0.13*ICa_1 - 0.075*Ca_1

    dVSdt_2 = 1.0/Cm * (IS_2/p     + IDS_2/p     - INa_2 - IK_2  - ISL_2   - I12)
    dVDdt_2 = 1.0/Cm * (ID_2/(1-p) - IDS_2/(1-p) - ICa_2 - IKC_2 - IKAHP_2 - IDL_2)
    dwdt_2  = phiw * (winf_2-w_2)/tauw_2
    dndt_2  = (ninf_2-n_2)/taun
    dhdt_2  = (hinf_2-h_2)/tauh
    dcdt_2  = alphac_2*(1-c_2) - betac_2*c_2
    dqdt_2  = alphaq_2*(1-q_2) - betaq_2*q_2
    dCadt_2 = -0.13*ICa_2 - 0.075*Ca_2

    return [dVSdt_1, dVDdt_1, dwdt_1, dndt_1, dhdt_1, dcdt_1, dqdt_1, dCadt_1,
            dVSdt_2, dVDdt_2, dwdt_2, dndt_2, dhdt_2, dcdt_2, dqdt_2, dCadt_2]


def dydt_vectorized(t, y, Data):
    # t.shape == (n,)
    # y.shape == (16,n)
    # Data is a dictionary with all the parameters
    
    dVSdt_1, dVDdt_1, dwdt_1, dndt_1, dhdt_1, dcdt_1, dqdt_1, dCadt_1, \
    dVSdt_2, dVDdt_2, dwdt_2, dndt_2, dhdt_2, dcdt_2, dqdt_2, dCadt_2 = \
        np.array([dydt(t[i], y[:, i], Data) for i in range(t.shape[0])]).T
    return dVSdt_1, dVDdt_1, dwdt_1, dndt_1, dhdt_1, dcdt_1, dqdt_1, dCadt_1, \
            dVSdt_2, dVDdt_2, dwdt_2, dndt_2, dhdt_2, dcdt_2, dqdt_2, dCadt_2