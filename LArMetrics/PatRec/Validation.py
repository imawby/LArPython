# Imports
import argparse
import uproot
import numpy as np
import awkward as ak
import os

import Definitions
import Variables
import EventValidationFunc
import HierarchyValidationFunc
import PFPValidationFunc
import ShowerValidationFunc
import TrackValidationFunc

def main(args) :

    #########################
    # Handle file
    #########################
    file_name = f'{args.validation_file}'
    file = uproot.open(file_name)

    #########################
    # Get branches
    #########################
    event_tree = file['EventTree']
    pfp_tree = file['PFPTree']
    track_tree = file['TrackTree']
    shower_tree = file['ShowerTree']
    hierarchy_tree = file['HierarchyTree']

    event_branches = event_tree.arrays(['Run', 'Subrun', 'Event', 'MCInt_IsCC', 'MCNu_PDG', 'MCNu_Energy', 'MCNu_VisEnergy',
                                        'MCNu_VertexX', 'MCNu_VertexY', 'MCNu_VertexZ',
                                        'RecoNu_VertexX', 'RecoNu_VertexY', 'RecoNu_VertexZ',
                                        'RecoNu_VertexAcc_Pass2'], library="ak")

    pfp_branches = pfp_tree.arrays(['MCP_TruePDG', 'BM_VertexAcc', 'MCP_TrueVisEnergy', 
                                    'MCP_NMCHits2D', 'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                    'BM_IsTrack', 'BM_IsShower',
                                    'BM_Completeness', 'BM_Purity'], library="ak")

    pfp_branches = pfp_tree.arrays(['MCP_TruePDG', 'MCP_TrueEnergy', 'MCP_TrueVisEnergy', 'MCP_TrueThetaXZ', 'MCP_TrueThetaYZ',
                                    'MCP_NMCHits2D', 'MCP_NMCHitsU', 'MCP_NMCHitsV', 'MCP_NMCHitsW',
                                    'MCP_HasMatch', 'MCP_Length', 'MCP_Displacement',
                                    'BM_IsTrack', 'BM_IsShower',
                                    'BM_Completeness', 'BM_CompletenessU', 'BM_CompletenessV', 'BM_CompletenessW',
                                    'BM_Purity', 'BM_PurityU', 'BM_PurityV', 'BM_PurityW',
                                    'BM_VertexAcc', 'BM_Length', 'BM_Displacement',
                                    'ALT_Completeness', 'ALT_Purity', 'ALT_PDG', 'ALT_IsUpstreamHierarchy', 'ALT_IsSameMC'], library="ak")    

    shower_branches = shower_tree.arrays(['MCP_TrueCoreLengthFromU', 'MCP_TrueCoreLengthFromV', 'MCP_TrueCoreLengthFromW',
                                          'BM_RecoCoreLength', 'BM_RecoLength', 'BM_MoliereRadius',
                                          'BM_DirAcc',
                                          'MCP_InitialMCHits', 'MCP_InitialMCHitsU', 'MCP_InitialMCHitsV', 'MCP_InitialMCHitsW',
                                          'BM_InitialPfoHits', 'BM_InitialPfoHitsU', 'BM_InitialPfoHitsV', 'BM_InitialPfoHitsW',
                                          'BM_InitialCompleteness', 'BM_InitialCompletenessU', 'BM_InitialCompletenessV', 'BM_InitialCompletenessW',
                                          'BM_InitialPurity', 'BM_InitialPurityU', 'BM_InitialPurityV', 'BM_InitialPurityW'], library="ak")    

    track_branches = track_tree.arrays(['BM_EndpointAcc', 'BM_EndpointCompleteness', 'BM_EndpointPurity', 
                                        'MCP_HasMichel', 'MCP_HasTargetMichel', 'BM_IsMichelRecod', 'MCP_MichelIndex', 
                                        'BM_MichelIsChild', 'BM_MichelIsShower', 'MCP_MichelFromMuon', 'MCP_EndpointsMCHits'], library="ak")    

    hierarchy_branches = hierarchy_tree.arrays(['MC_HierarchyTier', 'MC_ParentIndex', 'BM_HierarchyTier', 'BM_ParentIndex'], library="ak")

    ###################################
    # Create interaction/tier/PDG masks
    ###################################
    int_masks = Definitions.GetIntMasks(event_branches, pfp_branches={}, broadcast=False)
    int_masks_broadcast = Definitions.GetIntMasks(event_branches, pfp_branches)
    pdg_masks = Definitions.GetPDGMasks(pfp_branches)
    tier_masks = Definitions.GetTierMasks(hierarchy_branches)

    ##########################################################################################################
    # Implement custom definitions of reconstructable and reconstructed, to be used in correctness definitions
    ##########################################################################################################
    # Apply custom def of reconstructable and reconstructed
    # The tree constains some particles that we reconstructed, even if not initially deemed to be a target
    pfp_target_mask = Definitions.GetIsTargetMask(pfp_branches)
    pfp_reco_mask = Definitions.GetIsRecoMask(pfp_target_mask, pfp_branches)

    # Event Validation Plots
    event_plot_dir = f'{args.plot_dir}/EventValidation/'
    EventValidationFunc.run_event_validation(event_plot_dir, int_masks, event_branches)

    # Hierarchy Validation Plots
    hierarchy_plot_dir = f'{args.plot_dir}/HierarchyValidation/'
    HierarchyValidationFunc.run_hierarchy_validation(hierarchy_plot_dir, pfp_target_mask, pfp_reco_mask, int_masks_broadcast, tier_masks, pdg_masks, hierarchy_branches, pfp_branches)

    # PFP Validation Plots
    pfp_plot_dir = f'{args.plot_dir}/PFPValidation/'
    PFPValidationFunc.run_pfp_validation(pfp_plot_dir, pfp_target_mask, pfp_reco_mask, int_masks, tier_masks, pdg_masks, pfp_branches)

    # Shower Validation Plots
    shower_plot_dir = f'{args.plot_dir}/ShowerValidation/'
    ShowerValidationFunc.run_shower_validation(shower_plot_dir, pfp_target_mask, pfp_reco_mask, int_masks_broadcast, tier_masks, pdg_masks, shower_branches)

    # Track Validation Plots
    track_plot_dir = f'{args.plot_dir}/TrackValidation/Track'
    TrackValidationFunc.run_track_validation(track_plot_dir, pfp_target_mask, pfp_reco_mask, int_masks_broadcast, tier_masks, pdg_masks, track_branches)
    for michel_type in Definitions.michel_types :
            michel_plot_dir = f'{args.plot_dir}/TrackValidation/Michel/{Definitions.michel_type_strings[michel_type]}'
            TrackValidationFunc.run_michel_validation(michel_plot_dir, pfp_target_mask, pfp_reco_mask, int_masks_broadcast, tier_masks, pdg_masks, hierarchy_branches, pfp_branches, track_branches, michel_type)
    
##########################################################################################################
##########################################################################################################
    
def create_directory_structure(plot_dir) :
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    # EventTree
    create_tree_directory(plot_dir, 'EventValidation')
    create_subdirectory(f'{plot_dir}/EventValidation', 'MC', Variables.Event_MCP_plotting_vars)
    create_subdirectory(f'{plot_dir}/EventValidation', 'Reco', Variables.Event_Reco_plotting_vars)
    create_subdirectory(f'{plot_dir}/EventValidation', 'Diff', Variables.Event_diff_plotting_vars + [Variables.vtx_dr_all, Variables.vtx_dr_only_reco])
    
    # HierarchyTree
    create_tree_directory(plot_dir, 'HierarchyValidation')

    # PFPTree
    create_tree_directory(plot_dir, 'PFPValidation')
    create_tree_directory(f'{plot_dir}/PFPValidation', 'Efficiency')
    create_subdirectory(f'{plot_dir}/PFPValidation', 'MC', Variables.PFP_MCP_plotting_vars)
    create_subdirectory(f'{plot_dir}/PFPValidation', 'BM', Variables.PFP_BM_plotting_vars)
    create_subdirectory(f'{plot_dir}/PFPValidation', 'Diff', Variables.PFP_diff_plotting_vars)
    create_subdirectory(f'{plot_dir}/PFPValidation', 'Alt', Variables.PFP_ALT_plotting_vars)
    for plot_var in Variables.PFP_ALT_plotting_vars :
        create_subdirectory(f'{plot_dir}/PFPValidation/Alt', f'{plot_var.dir_name}_Seg', Variables.ALT_seg_vars)
    
    create_subdirectory(f'{plot_dir}/PFPValidation', 'XProfile', Variables.PFP_profile_vars)    
    create_subdirectory(f'{plot_dir}/PFPValidation', 'TrackShower', Variables.PFP_track_shower_plotting_vars)
    create_subdirectory(f'{plot_dir}/PFPValidation', 'Efficiency', Variables.PFP_efficiency_vars)
    create_subdirectory(f'{plot_dir}/PFPValidation', '2D', Variables.PFP_2D_vars)    

    # ShowerTree
    create_tree_directory(plot_dir, 'ShowerValidation')
    create_subdirectory(f'{plot_dir}/ShowerValidation', 'MC', Variables.Shower_MCP_plotting_vars)
    create_subdirectory(f'{plot_dir}/ShowerValidation', 'BM', Variables.Shower_BM_plotting_vars)
    create_subdirectory(f'{plot_dir}/ShowerValidation', 'Diff', Variables.Shower_diff_plotting_vars)

    # TrackTree
    create_tree_directory(plot_dir, 'TrackValidation')    
    create_tree_directory(f'{plot_dir}/TrackValidation', 'Track')
    create_subdirectory(f'{plot_dir}/TrackValidation/Track', 'MC', Variables.Track_MCP_plotting_vars)
    create_subdirectory(f'{plot_dir}/TrackValidation/Track', 'BM', Variables.Track_BM_plotting_vars)
    create_tree_directory(f'{plot_dir}/TrackValidation', 'Michel')
    for michel_type in Definitions.michel_types :
        create_tree_directory(f'{plot_dir}/TrackValidation/Michel', Definitions.michel_type_strings[michel_type])
        create_tree_directory(f'{plot_dir}/TrackValidation/Michel/{Definitions.michel_type_strings[michel_type]}', 'Efficiency')
        create_tree_directory(f'{plot_dir}/TrackValidation/Michel/{Definitions.michel_type_strings[michel_type]}', 'Hierarchy')
        create_subdirectory(f'{plot_dir}/TrackValidation/Michel/{Definitions.michel_type_strings[michel_type]}', 'MC', Variables.Michel_MCP_plotting_vars)
        create_subdirectory(f'{plot_dir}/TrackValidation/Michel/{Definitions.michel_type_strings[michel_type]}', 'TrackShower', Variables.Michel_track_shower_vars)
        create_subdirectory(f'{plot_dir}/TrackValidation/Michel/{Definitions.michel_type_strings[michel_type]}', 'Efficiency', Variables.Michel_efficiency_vars)

    
##########################################################################################################
##########################################################################################################
    
def create_tree_directory(root_dir, tree_name) :
    if not os.path.isdir(f'{root_dir}/{tree_name}') :
        os.makedirs(f'{root_dir}/{tree_name}')
    
##########################################################################################################
##########################################################################################################
    
def create_subdirectory(root_dir, sub_dir, plot_vars) :
    if not os.path.isdir(f'{root_dir}/{sub_dir}') :
        os.makedirs(f'{root_dir}/{sub_dir}/')
    for plot_var in plot_vars :
        if not os.path.isdir(f'{root_dir}/{sub_dir}/{plot_var.dir_name}') :
            os.makedirs(f'{root_dir}/{sub_dir}/{plot_var.dir_name}')
            
##########################################################################################################
##########################################################################################################
            
def parse_cli():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validation script for Pandora.")
    parser.add_argument("--plot_dir", type=str, required=True, help="Directory for storing plots.")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the validation file.")
    return parser.parse_args()

##########################################################################################################
##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    create_directory_structure(args.plot_dir)
    main(args)
