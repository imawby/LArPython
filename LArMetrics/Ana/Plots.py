import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

import Signal
import Selection

#####################################################################################################################
#####################################################################################################################

N_E_BINS = 32
N_E_RANGE = [0.0, 8.0]

#####################################################################################################################
#####################################################################################################################

class_strings = ['CCnue', 'CCnue_OutOfFV', 'CCnumu', 'CCnumu_OutOfFV', 'CCnutau', 'NC', 'Other']
class_style = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'solid', 'solid']
class_colour = ['red', 'red', 'blue', 'blue', 'green', 'gray', 'violet']

#####################################################################################################################
#####################################################################################################################

def save_plot(fig, path) :
    plt.close(fig)      
    fig.savefig(path, bbox_inches='tight')

#####################################################################################################################
#####################################################################################################################

class PlotVar :
    def __init__(self, var_name, input_type_name, x_label, range, n_bins):
        self.var_name = var_name
        self.input_type_name = input_type_name
        self.x_label = x_label
        self.range = range
        self.n_bins = n_bins
    
#####################################################################################################################
#####################################################################################################################

def PlotVariable(input_array, plot_var, ax, color, label, show_under_over_flow=False) :
    if (show_under_over_flow) :
        input_array[input_array < plot_var.range[0]] = plot_var.range[0]
        input_array[input_array > plot_var.range[1]] = plot_var.range[1]
    
    n_entries = input_array.shape[0]
    hist_counts, bin_edges = np.histogram(input_array, bins=plot_var.n_bins, range=plot_var.range)
    hist_fraction = hist_counts / n_entries
    
    # Plot with error == sqrt(n_i)/N
    hist_error = np.sqrt(hist_counts) / n_entries
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])    
    ax.step(bin_centers, hist_fraction, where='mid', color=color, linewidth=1, label=f'{label}')
    
    ax.fill_between(bin_centers,hist_fraction,step='mid',color=color,alpha=0.3)
    ax.errorbar(bin_centers, hist_fraction, yerr=hist_error, fmt='none', ecolor=color, capsize=2)
    ax.legend()

#####################################################################################################################
#####################################################################################################################

def PlotSignalBackgroundVar(flattened_branch, signal_mask, background_mask, plot_var, ax, x_label='', title='', show_under_over_flow=False) :
    signal_var = flattened_branch[signal_mask]
    background_var = flattened_branch[background_mask]

    PlotVariable(signal_var, plot_var, ax, 'blue', 'signal', show_under_over_flow)
    PlotVariable(background_var, plot_var, ax, 'red', 'background', show_under_over_flow)
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'Fraction of {plot_var.input_type_name}')
    ax.grid(True)
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True)
    ax.legend()    

#####################################################################################################################
#####################################################################################################################

def PlotEnergySpectrum(nusel_branches, tree_branch, target_mask, ax, color='black', legend='', style='solid') :
    target_entries = nusel_branches[tree_branch][target_mask]

    if target_entries.ndim > 1:
        target_entries = ak.flatten(target_entries)

    target_entries = ak.to_numpy(target_entries).reshape(-1)
    n_entries = target_entries.shape[0]
    
    if (n_entries == 0) : 
        return

    pot_weights = nusel_branches['ProjectedPOTWeight'][target_mask]
    osc_weights = nusel_branches['OscProb'][:, 4]
    osc_weights = ak.where(nusel_branches['NC'] == 1, 1, osc_weights)
    osc_weights = osc_weights[target_mask]
    weights = osc_weights*pot_weights
    
    ax.hist(target_entries, weights=weights, bins=N_E_BINS, range=N_E_RANGE, histtype='step', color=color, linewidth=1, label=legend, linestyle=style) 
    
#####################################################################################################################
#####################################################################################################################

def PlotEnergySpectrumDecomposition(nusel_branches, tree_branch, class_masks, selected_mask, fig, ax,title=''):
    all_entries = []
    all_weights = []
    labels = []
    colors = []

    for class_index in range(len(class_strings)):
        class_mask = selected_mask & class_masks[class_index]
        target_entries = nusel_branches[tree_branch][class_mask]

        if target_entries.ndim > 1:
            target_entries = ak.flatten(target_entries)

        target_entries = ak.to_numpy(target_entries).reshape(-1)

        if len(target_entries) == 0:
            continue

        pot_weights = nusel_branches['ProjectedPOTWeight'][class_mask]
        osc_weights = nusel_branches['OscProb'][:, 4]
        osc_weights = ak.where(nusel_branches['NC'] == 1, 1, osc_weights)
        osc_weights = osc_weights[class_mask]
        weights = ak.to_numpy(osc_weights * pot_weights).reshape(-1)

        all_entries.append(target_entries)
        all_weights.append(weights)
        labels.append(class_strings[class_index])
        colors.append(class_colour[class_index])

    # Plot stacked histogram
    counts, bins, patches = ax.hist(
     all_entries,
     weights=all_weights,
     bins=N_E_BINS,
     range=N_E_RANGE,
     stacked=True,
     histtype='bar',
     color=colors,
     label=labels,
     edgecolor='none',
     alpha=0.75)

    counts = np.atleast_2d(counts)

    # Draw boundaries between stacked layers
    for i in range(counts.shape[0]):
        ax.step(bins[:-1],counts[i],where='post',color='black',linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel(f'{tree_branch} [GeV]')
    ax.set_ylabel('nEntries')
    ax.grid(True)
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True)
    ax.legend()

#####################################################################################################################
#####################################################################################################################

def GetSelectionMetrics(nusel_branches, signal_mask, selected_mask) :
    pot_weights = nusel_branches['ProjectedPOTWeight']
    osc_weights = nusel_branches['OscProb'][:, 4]
    osc_weights = ak.where(nusel_branches['NC'] == 1, 1, osc_weights)
    weights = osc_weights*pot_weights

    total_sig = ak.sum(weights[signal_mask])
    total_sel = ak.sum(weights[selected_mask])
    total_sel_sig = ak.sum(weights[signal_mask & selected_mask])

    global_efficiency = float(total_sel_sig) / float(total_sig) if total_sig > 0.0001 else 0.0
    global_purity = float(total_sel_sig) / float(total_sel) if total_sel > 0.0001 else 0.0

    return {'Efficiency':global_efficiency, 'Purity':global_purity}
    

#####################################################################################################################
#####################################################################################################################

def PlotSelectionMetrics(nusel_branches, signal_mask, selected_mask, fig, ax, title='') :

    pot_weights = nusel_branches['ProjectedPOTWeight']
    osc_weights = nusel_branches['OscProb'][:, 4]
    osc_weights = ak.where(nusel_branches['NC'] == 1, 1, osc_weights)
    weights = osc_weights*pot_weights
    
    sig_entries = ak.to_numpy(nusel_branches['Enu'][signal_mask])
    sel_entries = ak.to_numpy(nusel_branches['Enu'][selected_mask])
    sel_sig_entries = ak.to_numpy(nusel_branches['Enu'][signal_mask & selected_mask])

    total_sig = ak.sum(weights[signal_mask])
    total_sel = ak.sum(weights[selected_mask])
    total_sel_sig = ak.sum(weights[signal_mask & selected_mask])
    
    global_efficiency = float(total_sel_sig) / (total_sig)
    global_purity = float(total_sel_sig) / (total_sel)
    print(title)
    print(f'global_efficiency: {round(global_efficiency, 2)* 100}%')
    print(f'global_purity: {round(global_purity, 2)* 100}%')

    hist_sig, edges = np.histogram(sig_entries, bins=N_E_BINS, range=N_E_RANGE, weights=ak.to_numpy(weights[signal_mask]))
    hist_sel, _ = np.histogram(sel_entries, bins=N_E_BINS, range=N_E_RANGE, weights=ak.to_numpy(weights[selected_mask]))
    hist_sel_sig, _ = np.histogram(sel_sig_entries, bins=N_E_BINS, range=N_E_RANGE, weights=ak.to_numpy(weights[signal_mask & selected_mask]))
    
    efficiency = np.divide(hist_sel_sig, hist_sig, out=np.zeros_like(hist_sel_sig, dtype=float), where=hist_sig > 0)
    purity = np.divide(hist_sel_sig, hist_sel, out=np.zeros_like(hist_sel_sig, dtype=float), where=hist_sel > 0)

    # Binomial efficiency uncertainty
    efficiency_err = np.zeros_like(efficiency)
    valid = hist_sig > 0
    efficiency_err[valid] = np.sqrt(
        efficiency[valid] * (1.0 - efficiency[valid]) / hist_sig[valid]
    )

    purity_err = np.zeros_like(purity)
    valid = (hist_sel > 0) & (abs(purity - 1.0) < 0.999)
    purity_err[valid] = np.sqrt(
        purity[valid] * (1.0 - purity[valid]) / hist_sel[valid]
    )

    # Let's plot
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.set_title(title)
    ax.errorbar(bin_centers, efficiency, yerr=efficiency_err, fmt='o', color='black', capsize=3, label='Efficiency')
    ax.errorbar(bin_centers, purity, yerr=purity_err, fmt='o', color='blue', capsize=3, label='Purity')
    ax.set_xlabel('True Nu Energy [GeV]')
    ax.set_ylabel('AU')
    ax.grid(True)
    
    # Add in underlying distribution 
    ax2 = ax.twinx()
    PlotEnergySpectrum(nusel_branches, 'Enu', signal_mask, ax2, color='gray', legend='Signal')
    ax2.set_ylabel('Count', color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax2.spines['right'].set_color('gray')
    
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2)    
