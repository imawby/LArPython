import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import Definitions
import Variables
import ValidationFunc

#####################################################################################################################################################
####################################################################################################################################################

def run_pfp_validation(plot_dir_path, config_target_mask, config_reco_mask, int_masks, tier_masks, pdg_masks, pfp_branches) :

    for tier in Definitions.tiers :     
        tier_mask = tier_masks[tier]
    
        # Efficiency file
        efficiency_file_name = f'EfficiencyTables_{Definitions.tier_strings[tier]}'
    
        with open(f"{plot_dir_path}/Efficiency/{efficiency_file_name}.txt", "w") as f_efficiency:
            for int_type in Definitions.ints :
                int_mask = int_masks[int_type]
        
                ValidationFunc.PrintEfficiencyTableHeader(int_type, f_efficiency)
                
                # Global track-shower confusion matrices
                fig, ax = plt.subplots()
                TrackShowerClassification(config_reco_mask, pdg_masks, pfp_branches, fig, ax, int_type, tier)
                plt.close(fig)
                file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}'
                fig.savefig(f'{plot_dir_path}/TrackShower/{file_name}.pdf', bbox_inches='tight')    
        
                for i_pdg in range(len(Definitions.pdgs)) :
                    pdg = Definitions.pdgs[i_pdg]
                    pdg_mask = pdg_masks[pdg]

                    # Plot_config
                    file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                    plot_config = ValidationFunc.PlotConfig(file_name, Definitions.int_strings[int_type], Definitions.tier_strings[tier], Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])

                    # Get masks
                    target_mask = tier_mask & int_mask & pdg_mask & config_target_mask
                    reco_mask = target_mask & config_reco_mask                    
        
                    # Efficiency Metrics
                    efficiency_metrics = ValidationFunc.CalculateEfficiencyMetrics(target_mask, reco_mask, True)
                    ValidationFunc.PrintEfficiencyTableEntry(tier, pdg, efficiency_metrics, f_efficiency)
                    
                    # Plot track/shower classifications
                    ValidationFunc.create_plots({'reco' : reco_mask}, pfp_branches, ValidationFunc.TrackShowerAsAFunctionOf, Variables.PFP_track_shower_plotting_vars, f'{plot_dir_path}/TrackShower', plot_config)
                    
                    # Plot MCP_var distributions
                    ValidationFunc.create_plots({'target' : target_mask}, pfp_branches, ValidationFunc.PlotVariable, Variables.PFP_MCP_plotting_vars, f'{plot_dir_path}/MC', plot_config)
        
                    # Plot BM_var distributions
                    ValidationFunc.create_plots({'reco' : reco_mask}, pfp_branches, ValidationFunc.PlotVariable, Variables.PFP_BM_plotting_vars, f'{plot_dir_path}/BM', plot_config)

                    # Plot alt vars
                    ValidationFunc.create_plots({'target' : target_mask}, pfp_branches, ValidationFunc.PlotVariable, Variables.PFP_ALT_plotting_vars, f'{plot_dir_path}/Alt', plot_config)

                    # Segment alt vars
                    ValidationFunc.segment_plot_vars({'target' : target_mask}, pfp_branches, ValidationFunc.SegmentAltVar, Variables.PFP_ALT_plotting_vars, Variables.ALT_seg_vars,
                                                     f'{plot_dir_path}/Alt', plot_config)

                    # Plot efficiency
                    ValidationFunc.create_plots({'target' : target_mask, 'reco' : reco_mask}, pfp_branches, ValidationFunc.PlotEfficiency, Variables.PFP_efficiency_vars, f'{plot_dir_path}/Efficiency', plot_config)
                    
                    # Plot diff_vars
                    ValidationFunc.create_plots({'reco' : reco_mask}, pfp_branches, ValidationFunc.PlotDiffVariable, Variables.PFP_diff_plotting_vars, f'{plot_dir_path}/Diff', plot_config)

                    # Plot Profiles
                    ValidationFunc.create_plots({'target' : target_mask}, pfp_branches, ValidationFunc.PlotProfileX, Variables.PFP_profile_vars, f'{plot_dir_path}/XProfile', plot_config)

                    # 2D_var distributions
                    ValidationFunc.create_plots({'target' : target_mask}, pfp_branches, ValidationFunc.Plot2DHist, Variables.PFP_2D_vars, f'{plot_dir_path}/2D', plot_config)
                                            
                ValidationFunc.PrintEfficiencyTableFooter(f_efficiency)

#####################################################################################################################################################
####################################################################################################################################################

def TrackShowerClassification(reco_mask, pdg_masks, pfp_branches, fig, ax, int_type, tier) :
    conf_matrix_eff = []

    for pdg in Definitions.pdgs :
        # Only look at those that have been reconstructed
        target_mask = pdg_masks[pdg] & reco_mask
        n_particle = ak.sum(target_mask)
        n_track = ak.sum(pfp_branches['BM_IsTrack'][target_mask] == 1)
        n_shower = ak.sum(pfp_branches['BM_IsShower'][target_mask] == 1)
        conf_matrix_eff.append([round(n_track / n_particle, 2), round(n_shower / n_particle, 2)])
        
    conf_matrix_eff = np.array(conf_matrix_eff)
    im = ax.imshow(conf_matrix_eff, cmap='Blues')
    # Axis ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Track", "Shower"])
    ax.set_yticks(range(len(Definitions.pdgs)))
    ax.set_yticklabels([str(p) for p in Definitions.pdgs])
    # Axis labels and title
    ax.set_xlabel("Reco Classification")
    ax.set_ylabel("True PDG")
    ax.set_title(f'{Definitions.int_strings[int_type]}: {Definitions.tier_strings[tier]}')    
    
    # Add text inside cells
    for i in range(conf_matrix_eff.shape[0]):
        for j in range(conf_matrix_eff.shape[1]):
            ax.text(j, i, conf_matrix_eff[i, j],
                    ha="center", va="center", color=("white" if conf_matrix_eff[i, j] > 0.5 else "black"))
    
    plt.tight_layout()
