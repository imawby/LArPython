import Definitions
import ValidationFunc
import Variables
        
#####################################################################################################################################################
####################################################################################################################################################

def run_shower_validation(plot_dir_path, config_target_mask, config_reco_mask, int_masks, tier_masks, pdg_masks, shower_branches) :

    for tier in Definitions.tiers :
        tier_mask = tier_masks[tier]
        
        for int_type in Definitions.ints :
            int_mask = int_masks[int_type]
    
            for i_pdg in range(len(Definitions.pdgs)) :
                pdg = Definitions.pdgs[i_pdg]
                pdg_mask = pdg_masks[pdg]
                
                # Plot_config
                file_name = f'{Definitions.int_file_strings[int_type]}_{Definitions.tier_strings[tier]}_{Definitions.pdg_strings[pdg]}'
                plot_config = ValidationFunc.PlotConfig(file_name, Definitions.int_strings[int_type], Definitions.tier_strings[tier], Definitions.pdg_strings[pdg], Definitions.pdg_color[pdg])
                
                # Get masks
                target_mask = config_target_mask & tier_mask & int_mask & pdg_mask
                reco_mask = target_mask & config_reco_mask

                # Plot MCP_var distributions
                ValidationFunc.create_plots({'target' : target_mask}, shower_branches, ValidationFunc.PlotVariable, Variables.Shower_MCP_plotting_vars, f'{plot_dir_path}/MC', plot_config)

                # Plot BM_var distributions
                ValidationFunc.create_plots({'reco' : reco_mask}, shower_branches, ValidationFunc.PlotVariable, Variables.Shower_BM_plotting_vars, f'{plot_dir_path}/BM', plot_config)

                # Plot diff_var distributions
                ValidationFunc.create_plots({'reco' : reco_mask}, shower_branches, ValidationFunc.PlotDiffVariable, Variables.Shower_diff_plotting_vars, f'{plot_dir_path}/Diff', plot_config)
