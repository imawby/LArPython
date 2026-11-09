import math
import matplotlib.pyplot as plt
import numpy as np

################################################################################################################################################

def PlotEfficiencyOverlay(data_1, label_1, data_2, label_2, title, location='lower right', filename='jam.png') : 
    fig, ax = plt.subplots()
    
    # Plot efficiency
    dist_color = 'blue'
    ax.errorbar(data_1['efficiency_x'], data_1['efficiency_y'], yerr=data_1['efficiency_y_err'],
                fmt='-', color=dist_color, capsize=3, label=label_1)
    dist_color = 'green'
    ax.errorbar(data_2['efficiency_x'], data_2['efficiency_y'], yerr=data_2['efficiency_y_err'],
                fmt='-', color=dist_color, capsize=3, label=label_2)    

    ax.set_xlabel(data_1['efficiency_x_label'])
    ax.set_ylabel(data_1['efficiency_y_label'])
    ax.set_ylim([0.0, 1.2])
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=location)
    
    # Add in underlying distribution 
    ax2 = ax.twinx()
    dist_color = 'black'
    ax2.step(data_1['dist_x'], data_1['dist_y'], where='mid', color=dist_color, linewidth=1, label=f'Dist')
    ax2.fill_between(data_1['dist_x'], data_1['dist_y'], step='mid', color=dist_color, alpha=0.3)
    ax2.errorbar(data_1['dist_x'], data_1['dist_y'], yerr=data_1['dist_y_err'], fmt='none', ecolor=dist_color, capsize=2)
    ax2.set_ylabel(data_1['dist_y_label'], color=dist_color)
    ax2.tick_params(axis='y', colors=dist_color)
    ax2.spines['right'].set_color(dist_color)

    # Create dictionary
    file_data = {'eff_1_x' : data_1['efficiency_x'], 'eff_1_y' : data_1['efficiency_y'], 'eff_1_y_err' : data_1['efficiency_y_err'],
                 'eff_2_x' : data_2['efficiency_x'], 'eff_2_y' : data_2['efficiency_y'], 'eff_2_y_err' : data_2['efficiency_y_err'],
                 'dist_1_x' : data_1['dist_x'], 'dist_1_y' : data_1['dist_y'], 'dist_1_y_eff' : data_1['dist_y_err'],
                 'dist_2_x' : data_2['dist_x'], 'dist_2_y' : data_2['dist_y'], 'dist_2_y_eff' : data_2['dist_y_err']}                 

    # Save
    fig.savefig(f'/Users/isobel/Desktop/LBLPlots/{filename}.png', format='png', bbox_inches='tight', dpi=300)
    np.savez(f'/Users/isobel/Desktop/LBLPlots/{filename}.npz', file_data)
    print('file name:', f'/Users/isobel/Desktop/LBLPlots/{filename}')
    plt.close(fig)    

################################################################################################################################################

def PlotProfileOverlay(data_1, label_1, data_2, label_2, title, location='lower right', filename='jam.png') : 
    fig, ax = plt.subplots()
    
    # Plot efficiency
    dist_color = 'blue'
    ax.errorbar(data_1['profile_x'], data_1['profile_y'], yerr=data_1['profile_y_err'],
                fmt='-', color=dist_color, capsize=3, label=label_1)
    dist_color = 'green'
    ax.errorbar(data_2['profile_x'], data_2['profile_y'], yerr=data_2['profile_y_err'],
                fmt='-', color=dist_color, capsize=3, label=label_2)    

    ax.set_xlabel(data_1['profile_x_label'])
    ax.set_ylabel(data_1['profile_y_label'])
    ax.set_ylim([0.0, 1.2])
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=location) 
    
    # Add in underlying distribution 
    ax2 = ax.twinx()
    dist_color = 'black'
    ax2.step(data_1['dist_x'], data_1['dist_y'], where='mid', color=dist_color, linewidth=1, label=f'Dist')
    ax2.fill_between(data_1['dist_x'], data_1['dist_y'], step='mid', color=dist_color, alpha=0.3)
    ax2.errorbar(data_1['dist_x'], data_1['dist_y'], yerr=data_1['dist_y_err'], fmt='none', ecolor=dist_color, capsize=2)
    ax2.set_ylabel(data_1['dist_y_label'], color=dist_color)
    ax2.tick_params(axis='y', colors=dist_color)
    ax2.spines['right'].set_color(dist_color)
    
    # Create dictionary
    file_data = {'profile_1_x' : data_1['profile_x'], 'profile_1_y' : data_1['profile_y'], 'profile_1_y_err' : data_1['profile_y_err'],
                 'profile_2_x' : data_2['profile_x'], 'profile_2_y' : data_2['profile_y'], 'profile_2_y_err' : data_2['profile_y_err'],
                 'dist_1_x' : data_1['dist_x'], 'dist_1_y' : data_1['dist_y'], 'dist_1_y_eff' : data_1['dist_y_err'],
                 'dist_2_x' : data_2['dist_x'], 'dist_2_y' : data_2['dist_y'], 'dist_2_y_eff' : data_2['dist_y_err']}

    # Save
    fig.savefig(f'/Users/isobel/Desktop/LBLPlots/{filename}.png', format='png', bbox_inches='tight', dpi=300)
    np.savez(f'/Users/isobel/Desktop/LBLPlots/{filename}.npz', file_data)
    print('filesname:', f'/Users/isobel/Desktop/LBLPlots/{filename}')
    plt.close(fig)         

################################################################################################################################################

def PlotDistOverlay(data_1, label_1, data_2, label_2, title, location='lower right', filename='jam.png') : 
    fig, ax = plt.subplots()
    
    dist_color = 'black'
    ax.step(data_1['dist_x'], data_1['dist_y'], where='mid', color=dist_color, linewidth=1, label=label_1, linestyle='--')
    if ('dist_y_err' in data_1) :
        ax.errorbar(data_1['dist_x'], data_1['dist_y'], yerr=data_1['dist_y_err'], fmt='none', ecolor=dist_color, capsize=2)
    dist_color = str(data_2['dist_color']) if str(data_2['dist_color']) != 'black' else 'green'
    ax.step(data_2['dist_x'], data_2['dist_y'], where='mid', color=dist_color, linewidth=1, label=label_2)
    if ('dist_y_err' in data_2) :
        ax.errorbar(data_2['dist_x'], data_2['dist_y'], yerr=data_2['dist_y_err'], fmt='none', ecolor=dist_color, capsize=2)    
    
    ax.grid(True)
    ax.set_xlabel(data_1['dist_x_label'])
    #ax.set_ylim([0,1])
    ax.set_ylabel(data_1['dist_y_label'])
    ax.set_title(title)
    ax.legend(loc=location)

    # Make dictionary
    file_data = {'dist_1_x' : data_1['dist_x'], 'dist_1_y' : data_1['dist_y'], 'dist_1_y_eff' : data_1['dist_y_err'],
                 'dist_2_x' : data_2['dist_x'], 'dist_2_y' : data_2['dist_y'], 'dist_2_y_eff' : data_2['dist_y_err']}   

    # Save
    fig.savefig(f'/Users/isobel/Desktop/LBLPlots/{filename}.png', format='png', bbox_inches='tight', dpi=300)
    np.savez(f'/Users/isobel/Desktop/LBLPlots/{filename}.npz', file_data)    
    print('filesname:', f'/Users/isobel/Desktop/LBLPlots/{filename}')
    plt.close(fig)     

################################################################################################################################################

def Plot2DHist(data, title, file_name) :
    fig, ax = plt.subplots()

    # Plot as colored mesh, transpose needed for correct orientation
    mesh = ax.pcolormesh(data['2d_dist_x'], data['2d_dist_y'], data['2d_dist_entries'], cmap='Blues')
    mesh.set_clim(0.0, 0.07) 
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(data['2d_dist_z_label'])
    ax.set_xlabel(data['2d_dist_x_label'])
    ax.set_ylabel(data['2d_dist_y_label'])
    ax.set_title(title, fontsize=14, pad=15)
    
    for i in range(len(data['2d_dist_x']) - 1):
        for j in range(len(data['2d_dist_y']) - 1):
            value = data['2d_dist_entries'][j, i]

            # Compute bin centers
            x_center = 0.5 * (data['2d_dist_x'][i] + data['2d_dist_x'][i + 1])
            y_center = 0.5 * (data['2d_dist_y'][j] + data['2d_dist_y'][j + 1])

            # Choose text color based on value for readability
            text_color = 'white' if value > 0.05 else 'black'

            ax.text(x_center, y_center, f"{value:.3f}", ha='center', va='center', fontsize=8, color=text_color)
            
    # Create dictionary
    file_data = {'2d_dist_x' : data['2d_dist_x'], '2d_dist_y' : data['2d_dist_y'], '2d_dist_entries' : data['2d_dist_entries'].T}   

    # Save
    fig.savefig(f'/Users/isobel/Desktop/LBLPlots/{file_name}.png', format='png', bbox_inches='tight', dpi=300)
    np.savez(f'/Users/isobel/Desktop/LBLPlots/{file_name}.npz', file_data)    
    print('filesname:', f'/Users/isobel/Desktop/LBLPlots/{file_name}')
    plt.close(fig)   
