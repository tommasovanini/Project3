import matplotlib.pyplot as plt


def plot_solutions(all_data, g_values, colors):
    
    time = all_data['time']

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VS_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{\mathrm{S}, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 2)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VD_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{\mathrm{D}, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 3)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['ICa_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$I_{\mathrm{Ca}, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 4)
    for i, g_val in enumerate(g_values):
        plt.plot(time, -all_data['IDS_1'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$-I_{\mathrm{DS}, 1}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VS_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{\mathrm{S}, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 2)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['VD_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$V_{\mathrm{D}, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 3)
    for i, g_val in enumerate(g_values):
        plt.plot(time, all_data['ICa_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$I_{\mathrm{Ca}, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.subplot(2, 2, 4)
    for i, g_val in enumerate(g_values):
        plt.plot(time, -all_data['IDS_2'][i], color=colors[i], label=f'g={g_val}')
    plt.title('$-I_{\mathrm{DS}, 2}$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.show()



    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(121, projection='3d')
    for i, g_val in enumerate(g_values):
        ax.plot(time, all_data['VS_1'][i], all_data['w_1'][i], color=colors[i], label=f'g={g_val}')
    ax.view_init(10, 260)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('$V_{\mathrm{S}, 1}$ (mV)')
    ax.set_zlabel('$w_1$')
    plt.title("t vs $V_{\mathrm{S}, 1}$ vs $w_1$")
    plt.legend(bbox_to_anchor=(1.01, 1))

    ax = fig.add_subplot(122, projection='3d')
    for i, g_val in enumerate(g_values):
        ax.plot(time, all_data['VS_2'][i], all_data['w_2'][i], color=colors[i], label=f'g={g_val}')
    ax.view_init(10, 260)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('$V_{\mathrm{S}, 2}$ (mV)')
    ax.set_zlabel('$w_2$')
    plt.title("t vs $V_{\mathrm{S}, 2}$ vs $w_2$")
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.show()



    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(121, projection='3d')
    for i, g_val in enumerate(g_values):
        ax.plot(all_data['ICa_1'][i], all_data['VS_1'][i], all_data['VD_1'][i], color=colors[i], label=f'g={g_val}')
    ax.view_init(10, 260)
    ax.set_xlabel('$I_{\mathrm{Ca}, 1}$ ($\mu A/cm^2$)')
    ax.set_ylabel('$V_{\mathrm{S}, 1}$ (mV)')
    ax.set_zlabel('$V_{\mathrm{D}, 1}$ (mV)')
    plt.title("$I_{\mathrm{Ca}, 1}$ vs $V_{\mathrm{S}, 1}$ vs $V_{\mathrm{D}, 1}$")
    plt.legend(bbox_to_anchor=(1.01, 1))

    ax = fig.add_subplot(122, projection='3d')
    for i, g_val in enumerate(g_values):
        ax.plot(all_data['ICa_2'][i], all_data['VS_2'][i], all_data['VD_2'][i], color=colors[i], label=f'g={g_val}')
    ax.view_init(10, 260)
    ax.set_xlabel('$I_{\mathrm{Ca}, 2}$ ($\mu A/cm^2$)')
    ax.set_ylabel('$V_{\mathrm{S}, 2}$ (mV)')
    ax.set_zlabel('$V_{\mathrm{D}, 2}$ (mV)')
    plt.title("$I_{\mathrm{Ca}, 2}$ vs $V_{\mathrm{S}, 2}$ vs $V_{\mathrm{D}, 2}$")
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.show()