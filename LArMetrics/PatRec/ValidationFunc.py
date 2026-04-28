import awkward as ak
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import Definitions

##############################################################################################
##############################################################################################
    
class PlotConfig :
    def __init__(self, file_name, int_string, tier_string, pdg_string, color) :
        self.file_name = file_name
        self.int_string = int_string
        self.tier_string = tier_string        
        self.pdg_string = pdg_string
        self.color = color

##############################################################################################
##############################################################################################

def configure_plot(fig, ax, int_string="", tier_string="", pdg_string="") :

    title = '       '
    if (int_string) :
        title += int_string
    if (tier_string) :
        title += f' - {tier_string}'
    if (pdg_string) :
        title += f' - {pdg_string}'    
    ax.set_title(title)
    ax.grid(True)
    ax.tick_params(labelbottom=True, bottom=True, labelleft=True, left=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95, hspace=0.4, wspace=0.4)

##############################################################################################
##############################################################################################

def save_plot(fig, path) :
    plt.close(fig)      
    fig.savefig(path, bbox_inches='tight')

##############################################################################################
##############################################################################################    

def create_plots(masks, branches, plot_func, plot_vars, sub_dir, plot_config) :
    for plot_var in plot_vars :
        fig, ax = plt.subplots()
        configure_plot(fig, ax, int_string=plot_config.int_string, pdg_string=plot_config.pdg_string)
        
        if len(masks) == 1 :
            plot_func(next(iter(masks.values())), branches, plot_var, ax, plot_config.pdg_string, plot_config.color)
        else :
            plot_func(masks['target'], masks['reco'], branches, plot_var, ax, plot_config.pdg_string, plot_config.color)
            
        save_plot(fig, f'{sub_dir}/{plot_var.dir_name}/{plot_config.file_name}.pdf')


##############################################################################################
##############################################################################################    

def segment_plot_vars(masks, branches, plot_func, plot_vars, seg_vars, sub_dir, plot_config) :

    for plot_var in plot_vars :
        for seg_var in seg_vars :
            fig, ax = plt.subplots()
            configure_plot(fig, ax, int_string=plot_config.int_string, pdg_string=plot_config.pdg_string)

            if len(masks) == 1 :
                plot_func(next(iter(masks.values())), branches, plot_var, seg_var, ax)

            save_plot(fig, f'{sub_dir}/{plot_var.dir_name}_Seg/{seg_var.dir_name}/{plot_config.file_name}.pdf')

#####################################################################################################################################################
#####################################################################################################################################################

def SegmentAltVar(target_mask, pfp_branches, plot_var, seg_var, ax):

    n_entries = ak.count_nonzero(target_mask)
    
    for index in range(len(seg_var.options)) :
        mask = target_mask & (pfp_branches[seg_var.tree_name] == seg_var.options[index])
        PlotVariable(mask, pfp_branches, plot_var, ax, f'{seg_var.options[index]}', seg_var.colors[index], fill=False, n_entries=n_entries)

##############################################################################################
##############################################################################################

def TrackShowerAsAFunctionOf(pfp_indices, pfp_branches, plot_var, ax, legend_string, color) :
    
    n_hits = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][pfp_indices]))
    is_track = ak.to_numpy(ak.flatten(pfp_branches['BM_IsTrack'][pfp_indices]))
    is_shower = ak.to_numpy(ak.flatten(pfp_branches['BM_IsShower'][pfp_indices]))

    hist_all, edges = np.histogram(n_hits, bins=plot_var.n_bins, range=plot_var.range)
    hist_track, _ = np.histogram(n_hits[is_track == 1], bins=plot_var.n_bins, range=plot_var.range)
    hist_shower, _ = np.histogram(n_hits[is_shower == 1], bins=plot_var.n_bins, range=plot_var.range)
    
    proportion_track = np.divide(hist_track, hist_all, out=np.zeros_like(hist_track, dtype=float), where=hist_all > 0)
    proportion_shower = np.divide(hist_shower, hist_all, out=np.zeros_like(hist_shower, dtype=float), where=hist_all > 0)
    err_track = np.sqrt(proportion_track * (1.0 - proportion_track) / np.maximum(hist_all, 1))
    err_shower = np.sqrt(proportion_shower * (1.0 - proportion_shower) / np.maximum(hist_all, 1))
    
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(bin_centers, proportion_shower, yerr=err_shower, marker='x', capsize=2, label=(f'{legend_string} - Shower'))
    ax.errorbar(bin_centers, proportion_track, yerr=err_track, marker='x', capsize=2, label=(f'{legend_string} - Track'))

    ax.set_xlabel(plot_var.x_label)
    ax.set_ylabel(plot_var.y_label)    
    ax.legend(loc='center right')
    
##############################################################################################
##############################################################################################

def PlotVariable(indices_or_mask, branches, plot_var, ax, label, color, fill=True, n_entries=0):
    target_entries = branches[plot_var.tree_name][indices_or_mask]

    if target_entries.ndim > 1:
        target_entries = ak.flatten(target_entries)

    target_entries = ak.to_numpy(target_entries).reshape(-1)
    
    if (n_entries == 0) : 
        n_entries = len(target_entries)        
        if (n_entries == 0) :
            return
    
    hist_counts, bin_edges = np.histogram(
        target_entries, bins=plot_var.n_bins, range=plot_var.range
    )
    hist_fraction = hist_counts / n_entries
    
    # Plot with error == sqrt(n_i)/N
    hist_error = np.sqrt(hist_counts) / n_entries
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])    
    ax.step(bin_centers, hist_fraction, where='mid', color=color, linewidth=1, label=f'{label}')
    if (fill) :
        ax.fill_between(bin_centers,hist_fraction,step='mid',color=color,alpha=0.3)
    ax.errorbar(bin_centers, hist_fraction, yerr=hist_error, fmt='none', ecolor=color, capsize=2)

    ax.set_xlabel(plot_var.x_label)
    ax.set_ylabel(plot_var.y_label)

    return [hist_fraction, hist_error, bin_edges]

##############################################################################################
##############################################################################################

def Plot2DHist(indices_or_mask, branches, two_plot_var, ax, label, color) :
    plot_var_x = two_plot_var.plot_var_x
    plot_var_y = two_plot_var.plot_var_y
    target_x_entries = branches[plot_var_x.tree_name][indices_or_mask]
    target_y_entries = branches[plot_var_y.tree_name][indices_or_mask]

    if target_x_entries.ndim > 1:
        target_x_entries = ak.flatten(target_x_entries)

    if target_y_entries.ndim > 1:
        target_y_entries = ak.flatten(target_y_entries)        

    target_x_entries = ak.to_numpy(target_x_entries).reshape(-1)
    target_y_entries = ak.to_numpy(target_y_entries).reshape(-1)
    
    # Compute 2D histogram
    hist_counts, x_edges, y_edges = np.histogram2d(
        target_x_entries,
        target_y_entries,
        bins=[plot_var_x.n_bins, plot_var_y.n_bins],
        range=[plot_var_x.range, plot_var_y.range]
    )

    n_entries = len(target_x_entries)
    
    if (n_entries == 0) :
        return
    
    hist_fraction = hist_counts / n_entries

    # Plot as colored mesh
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        hist_fraction.T,   # transpose needed for correct orientation
        cmap='Blues'
    )
    mesh.set_clim(0.0, 0.1) 
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(plot_var_x.y_label)
    ax.set_xlabel(plot_var_x.x_label)
    ax.set_ylabel(plot_var_y.x_label)
    
    # Annotate each bin with its fraction value
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            value = hist_fraction[i, j]
    
            # Optional: skip empty bins
            if hist_counts[i, j] == 0:
                continue
    
            # Compute bin centers
            x_center = 0.5 * (x_edges[i] + x_edges[i + 1])
            y_center = 0.5 * (y_edges[j] + y_edges[j + 1])
    
            # Choose text color based on value for readability
            color = 'white' if value > 0.05 else 'black'
    
            ax.text(
                x_center,
                y_center,
                f"{value:.3f}",   # format (3 decimal places)
                ha='center',
                va='center',
                fontsize=8,
                color=color
            )   

##############################################################################################
##############################################################################################

def PlotProfileX(indices_or_mask, branches, profile_var, ax, label, color):
    target_x_entries = branches[profile_var.plot_var_x.tree_name][indices_or_mask]
    target_y_entries = branches[profile_var.plot_var_y.tree_name][indices_or_mask]

    if target_x_entries.ndim > 1:
        target_x_entries = ak.flatten(target_x_entries)

    if target_y_entries.ndim > 1:
        target_y_entries = ak.flatten(target_y_entries)

    # Mean
    mean_y, x_edges, _ = binned_statistic(target_x_entries, target_y_entries,
        statistic='mean', bins=profile_var.plot_var_x.n_bins, range=profile_var.plot_var_x.range)

    # Standard deviation
    std_y, _, _ = binned_statistic(target_x_entries, target_y_entries,
        statistic='std', bins=profile_var.plot_var_x.n_bins, range=profile_var.plot_var_x.range)

    # Counts per bin
    counts, _, _ = binned_statistic(target_x_entries, target_y_entries,
        statistic='count', bins=profile_var.plot_var_x.n_bins, range=profile_var.plot_var_x.range)

    # Standard error of the mean
    sem_y = std_y / np.sqrt(counts)

    # Bin centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    
    # Remove empty bins safely
    valid = counts > 0
    
    ax.errorbar(x_centers[valid], mean_y[valid], yerr=sem_y[valid], fmt='o',
        markersize=4, capsize=3, label=label, color='black')

    ax.set_ylabel(f"Mean {profile_var.plot_var_y.x_label}")
    ax.set_xlabel(profile_var.plot_var_x.x_label)
    ax.set_ylim(profile_var.plot_var_y.range)
    ax.grid(True)
    
    # Add in underlying distribution 
    ax2 = ax.twinx()
    PlotVariable(indices_or_mask, branches, profile_var.plot_var_x, ax2, 'Dist', color)
    ax2.set_ylabel(profile_var.plot_var_x.y_label, color=color)
    ax2.tick_params(axis='y', colors=color)
    ax2.spines['right'].set_color(color)
        
##############################################################################################
##############################################################################################

def PlotDiffVariable(indices_or_mask, branches, plot_diff_var, ax, label, color) :
    metric = branches[plot_diff_var.true_tree_name] - branches[plot_diff_var.reco_tree_name]
    target_entries = metric[indices_or_mask]
    
    if (target_entries.ndim > 1) :
        target_entries = ak.flatten(target_entries)
    
    target_entries = ak.to_numpy(target_entries).reshape(-1)
    n_target_entries = len(target_entries)

    if (n_target_entries == 0) :
        return
    
    weights = np.ones(n_target_entries) * (1.0 / n_target_entries)
    ax.set_xlabel(plot_diff_var.x_label)
    ax.set_ylabel(plot_diff_var.y_label)
    ax.hist(target_entries, bins=plot_diff_var.n_bins, range=plot_diff_var.range, weights=weights, histtype='step', color=color, linewidth=1, label=(f' {label} '))
    ax.legend()

##############################################################################################
##############################################################################################

def PlotEfficiency(target_mask_or_indices, reco_mask_or_indices, pfp_branches, plot_var, ax, legend_string, color) :
    
    target_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][target_mask_or_indices]))
    reco_entries = ak.to_numpy(ak.flatten(pfp_branches[plot_var.tree_name][reco_mask_or_indices]))
    
    hist_target, edges = np.histogram(target_entries, bins=plot_var.n_bins, range=plot_var.range)
    hist_reco, _ = np.histogram(reco_entries, bins=plot_var.n_bins, range=plot_var.range)
    efficiency = np.divide(hist_reco, hist_target, 
                           out=np.zeros_like(hist_reco, dtype=float), 
                           where=hist_target > 0)

    # Binomial efficiency uncertainty
    efficiency_err = np.zeros_like(efficiency)
    valid = hist_target > 0
    efficiency_err[valid] = np.sqrt(
        efficiency[valid] * (1.0 - efficiency[valid]) / hist_target[valid]
    )

    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    ax.errorbar(bin_centers, efficiency, yerr=efficiency_err, fmt='x-', color='black', capsize=3, label=f' {legend_string} ')
    ax.set_xlabel(plot_var.x_label)
    ax.set_ylabel('Efficiency')
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    
    # Add in underlying distribution 
    ax2 = ax.twinx()
    PlotVariable(target_mask_or_indices, pfp_branches, plot_var, ax2, 'Dist', color)
    ax2.set_ylabel(plot_var.y_label, color=color)
    ax2.tick_params(axis='y', colors=color)
    ax2.spines['right'].set_color(color)
    ax.legend(loc='center right')

    return [efficiency, efficiency_err, edges]

##############################################################################################
##############################################################################################

def Plot2DEfficiency(target_mask_or_indices, reco_mask_or_indices, branches, plot_var_x, plot_var_y, ax, label):
    target_x_entries = branches[plot_var_x.tree_name][target_mask_or_indices]
    target_y_entries = branches[plot_var_y.tree_name][target_mask_or_indices]
    reco_x_entries = branches[plot_var_x.tree_name][reco_mask_or_indices]
    reco_y_entries = branches[plot_var_y.tree_name][reco_mask_or_indices]

    if target_x_entries.ndim > 1:
        target_x_entries = ak.flatten(target_x_entries)
    if reco_x_entries.ndim > 1:
        reco_x_entries = ak.flatten(reco_x_entries)         
    if target_y_entries.ndim > 1:
        target_y_entries = ak.flatten(target_y_entries)
    if reco_y_entries.ndim > 1:
        reco_y_entries = ak.flatten(reco_y_entries)         

    target_x_entries = ak.to_numpy(target_x_entries).reshape(-1)
    target_y_entries = ak.to_numpy(target_y_entries).reshape(-1)
    reco_x_entries = ak.to_numpy(reco_x_entries).reshape(-1)
    reco_y_entries = ak.to_numpy(reco_y_entries).reshape(-1)    
    
    hist_target, x_edges, y_edges = np.histogram2d(
        target_x_entries, target_y_entries,
        bins=[plot_var_x.n_bins, plot_var_y.n_bins],
        range=[plot_var_x.range, plot_var_y.range]
    )
    hist_reco, _, _ = np.histogram2d(
        reco_x_entries, reco_y_entries,
        bins=[plot_var_x.n_bins, plot_var_y.n_bins],
        range=[plot_var_x.range, plot_var_y.range]
    )

    # Efficiency
    efficiency = np.divide(
        hist_reco, hist_target,
        out=np.zeros_like(hist_reco, dtype=float),
        where=hist_target > 0
    )

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        efficiency.T,   # transpose for correct orientation
        cmap='Blues',
        vmin=0, vmax=1
    )
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(plot_var_x.y_label)
    ax.set_xlabel(plot_var_x.x_label)
    ax.set_ylabel(plot_var_y.x_label)
    
    
##############################################################################################
##############################################################################################

def PrintHierarchyTableHeader(int_type, file) :
    print(f'{Definitions.int_strings[int_type]}', file=file)
    print('------------------------------------------------------------------------------------', file=file)
    print('           | Correct Parent | False Primary | Wrong Parent | Parent Not Best Match |', file=file)
    print('------------------------------------------------------------------------------------', file=file) 

##############################################################################################
##############################################################################################

def CalculateHierarchyMetrics(hierarchy_branches, reco_michel_indices) :
    mc_tier         = hierarchy_branches['MC_HierarchyTier']
    bm_tier         = hierarchy_branches['BM_HierarchyTier']
    mc_parent       = hierarchy_branches['MC_ParentIndex']
    bm_parent       = hierarchy_branches['BM_ParentIndex']

    mc_tier_michel      = ak.to_numpy(ak.flatten(mc_tier[reco_michel_indices]))
    bm_tier_michel      = ak.to_numpy(ak.flatten(bm_tier[reco_michel_indices]))
    mc_parent_michel    = ak.to_numpy(ak.flatten(mc_parent[reco_michel_indices]))
    bm_parent_michel    = ak.to_numpy(ak.flatten(bm_parent[reco_michel_indices]))

    n_michel = mc_tier_michel.shape[0]
    n_not_best_match = np.count_nonzero((bm_tier_michel != 1) & (bm_parent_michel == -1))
    n_false_primary  = np.count_nonzero(bm_tier_michel == 1)
    n_correct_parent = np.count_nonzero((bm_tier_michel != 1) & (bm_parent_michel == mc_parent_michel))
    n_false_parent = np.count_nonzero((bm_tier_michel != 1) & (bm_parent_michel != -1) & (bm_parent_michel != mc_parent_michel))

    hierarchy_metrics = {}
    hierarchy_metrics['frac_not_best_match'] = round(0.0 if n_michel == 0 else float(n_not_best_match) / float(n_michel), 2)
    hierarchy_metrics['frac_false_primary'] = round(0.0 if n_michel == 0 else float(n_false_primary) / float(n_michel), 2)
    hierarchy_metrics['frac_correct_parent'] = round(0.0 if n_michel == 0 else float(n_correct_parent) / float(n_michel), 2)
    hierarchy_metrics['frac_false_parent'] = round(0.0 if n_michel == 0 else float(n_false_parent) / float(n_michel), 2)
    return hierarchy_metrics

##############################################################################################
##############################################################################################

def PrintHierarchyTableEntry(tier, hierarchy_metrics, file) :
    print(' ' + str(Definitions.tier_strings[tier]) + str(' '* (10 - len(str(Definitions.tier_strings[tier])))) +
                                            '|' + str(hierarchy_metrics['frac_correct_parent']) + str(' '* (16 - len(str(hierarchy_metrics['frac_correct_parent'])))) + \
                                            '|' + str(hierarchy_metrics['frac_false_primary']) + str(' '* (15 - len(str(hierarchy_metrics['frac_false_primary'])))) + \
                                            '|' + str(hierarchy_metrics['frac_false_parent']) + str(' '* (14 - len(str(hierarchy_metrics['frac_false_parent'])))) + \
                                            '|' + str(hierarchy_metrics['frac_not_best_match']) + str(' '* (23 - len(str(hierarchy_metrics['frac_not_best_match'])))) + \
                                            '|', file=file)

##############################################################################################
##############################################################################################

def PrintHierarchyTableFooter(file) :
    print('------------------------------------------------------------------------------------', file=file)
    print('', file=file)

##############################################################################################
##############################################################################################

def PrintEfficiencyTableHeader(int_type, file) :
    print(f'{Definitions.int_strings[int_type]}', file=file)
    print('------------------------------------------------------------------------', file=file)
    print('                  |      NTarget      |       NReco       | Efficiency |', file=file)
    print('------------------------------------------------------------------------', file=file)     

##############################################################################################
##############################################################################################

def CalculateEfficiencyMetrics(target_mask_or_indices, reco_mask_or_indices, is_mask) :

    if (is_mask) :
        n_targets = ak.sum(target_mask_or_indices)
        n_reco = ak.sum(reco_mask_or_indices)
    else :
        n_targets = len(ak.flatten(target_mask_or_indices))
        n_reco = len(ak.flatten(reco_mask_or_indices))
    
    reco_efficiency = 0 if n_targets == 0 else round(float(n_reco) / n_targets, 2)    
    
    efficiency_metrics = {}
    efficiency_metrics['NTarget'] = n_targets
    efficiency_metrics['NReco'] = n_reco
    efficiency_metrics['Efficiency'] = reco_efficiency
    return efficiency_metrics

##############################################################################################
##############################################################################################

def PrintEfficiencyTableEntry(tier, pdg, efficiency_metrics, file) :
    title_string = f'{Definitions.tier_strings[tier]} {Definitions.pdg_strings[pdg]}'
    print(' ' + title_string + str(' '* (18 - len(title_string))) +
                                            '|' + str(efficiency_metrics['NTarget']) + str(' '* (19 - len(str(efficiency_metrics['NTarget'])))) + \
                                            '|' + str(efficiency_metrics['NReco']) + str(' '* (19 - len(str(efficiency_metrics['NReco'])))) + \
                                            '|' + str(efficiency_metrics['Efficiency']) + str(' '* (12 - len(str(efficiency_metrics['Efficiency'])))) + \
                                            '|', file=file)

##############################################################################################
##############################################################################################

def PrintEfficiencyTableFooter(file) :
    print('------------------------------------------------------------------------', file=file)
    print('', file=file)
