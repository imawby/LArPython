# Imports
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import os

import Signal
import Selection
import Plots
import Definitions

##########################################################################################################
##########################################################################################################
    
def ConvertMuonMomToEnergy(muon_mom_array) :
    muon_mass = 0.1056583745
    jam = np.sqrt((muon_mom_array * muon_mom_array) + (muon_mass * muon_mass))
    jam = ak.where(muon_mom_array < -990, -999.0, jam)
    return jam

##########################################################################################################
##########################################################################################################

def main(args) :

    #########################
    # Handle file
    #########################
    file_name = f'{args.input_file}'
    file = uproot.open(file_name)

    #########################
    # Get branches
    #########################
    tree = file['ccnuselection/ccnusel']
    nusel_branches = tree.arrays(['Run', 'SubRun', 'Event', 'BeamPdg', 'NuPdg', 'NC', 'TargetZ', 
                                  'NuX', 'NuY', 'NuZ', 'Enu', 'OscProb',
                                  'NumuRecoENu', 'NumuRecoMomLep', 'NumuRecoEHad', 'NueRecoENu', 'NueRecoEHad', 'RecoTrackRecoContained',
                                  'RecoNuVtxX', 'RecoNuVtxY', 'RecoNuVtxZ',
                                  'CVNResultNue', 'CVNResultNumu',
                                  'RecoPFPTruePDG', 'RecoPFPTrackShowerScore', 'RecoPFPRecoNHits',
                                  'RecoTrackPandizzleVar', 'SelTrackPandizzleIndex',
                                  'SelPandizzleTrackContained', 'SelPandizzleTrackRecoMom', 'SelPandizzleTrackNumuEnu', 'SelPandizzleTrackNumuEHad',
                                  'SelIvysaurusTrackContained', 'SelIvysaurusTrackRecoMom', 'SelIvysaurusTrackNumuEnu', 'SelIvysaurusTrackNumuEHad',
                                  'SelShowerEnhancedPandrizzleScore', 'SelShowerBackupPandrizzleScore',
                                  'SelPandrizzleShowerNueEnu', 'SelPandrizzleShowerNueEHad', 'SelShowerPandrizzleIndex',
                                  'SelTrackPandizzleScore', 'SelTrackIvysaurusScore', 'SelShowerIvysaurusScore', 'SelShowerIvysaurusIndex',
                                  'SelIvysaurusShowerNueEnu', 'SelIvysaurusShowerNueEHad',
                                  'RecoShowerEnhancedPandrizzleScore', 'RecoShowerBackupPandrizzleScore', 'SelShowerPandrizzleIndex', 
                                  'ProjectedPOTWeight'], library='ak')

    ##################################
    # Add selected PFP nHits2D to tree
    ##################################
    for pid_string in ['Pandrizzle', 'Ivysaurus'] :
        idx = nusel_branches[f'SelShower{pid_string}Index']
        idx = idx[:, None]   # ← this is the "unsqueeze"
        valid = idx != -1
        masked_idx = ak.mask(idx, valid)
        values = nusel_branches['RecoPFPRecoNHits'][masked_idx]
        sel_shower_hits = ak.fill_none(values, -1.0)
        sel_shower_hits = sel_shower_hits[:,0]

        nusel_branches = ak.with_field(
            nusel_branches,
            sel_shower_hits,
            f'SelShower{pid_string}NHits'
        )

    ##############################
    # Correct reco nue energy
    ##############################
    IntShwEnergy = Definitions.IntShwEnergy_HD if args.is_hd else Definitions.IntShwEnergy_VD
    GradShwEnergy = Definitions.GradShwEnergy_HD if args.is_hd else Definitions.GradShwEnergy_VD
    IntNuEHadEn = Definitions.IntNuEHadEn_HD if args.is_hd else Definitions.IntNuEHadEn_VD
    GradNuEHadEn = Definitions.GradNuEHadEn_HD if args.is_hd else Definitions.GradNuEHadEn_VD

    for pid_string in ['Pandrizzle', 'Ivysaurus'] :
        enu_string = f'Sel{pid_string}ShowerNueEnu'
        ehad_string = f'Sel{pid_string}ShowerNueEHad'

        nue_electron_corrected = nusel_branches[enu_string] - nusel_branches[ehad_string]
        nue_electron_corrected = ak.where(nusel_branches[enu_string] < -990, -999.0, (nue_electron_corrected - IntShwEnergy) / GradShwEnergy)
        # Correct hadron
        nue_had_corrected = nusel_branches[ehad_string]
        nue_had_corrected = ak.where(nusel_branches[enu_string] < -990, -999.0, (nue_had_corrected - IntNuEHadEn) / GradNuEHadEn)
        # Add
        nue_corrected = nue_electron_corrected + nue_had_corrected
        nue_corrected = ak.where(nusel_branches[enu_string] < -990, -999.0, nue_corrected)

        nusel_branches = ak.with_field(
            nusel_branches,
            nue_corrected,
            f'Corrected{pid_string}NueRecoE'
        )

    IntTrkMomRange = Definitions.IntTrkMomRange_HD if args.is_hd else Definitions.IntTrkMomRange_VD
    GradTrkMomRange = Definitions.GradTrkMomRange_HD if args.is_hd else Definitions.GradTrkMomRange_VD
    IntTrkMomMCS = Definitions.IntTrkMomMCS_HD if args.is_hd else Definitions.IntTrkMomMCS_VD
    GradTrkMomMCS = Definitions.GradTrkMomMCS_HD if args.is_hd else Definitions.GradTrkMomMCS_VD
    IntNuMuHadEnCont = Definitions.IntNuMuHadEnCont_HD if args.is_hd else Definitions.IntNuMuHadEnCont_VD
    GradNuMuHadEnCont = Definitions.GradNuMuHadEnCont_HD if args.is_hd else Definitions.GradNuMuHadEnCont_VD
    IntNuMuHadEnExit = Definitions.IntNuMuHadEnExit_HD if args.is_hd else Definitions.IntNuMuHadEnExit_VD
    GradNuMuHadEnExit = Definitions.GradNuMuHadEnExit_HD if args.is_hd else Definitions.GradNuMuHadEnExit_VD

    ##############################
    # Correct reco numu energy
    ##############################

    for pid_string in ['Pandizzle', 'Ivysaurus'] :
        enu_string = f'Sel{pid_string}TrackNumuEnu'
        mom_string = f'Sel{pid_string}TrackRecoMom'
        contained_string = f'Sel{pid_string}TrackContained'
        ehad_string = f'Sel{pid_string}TrackNumuEHad'

        # Correct muon
        numu_muon_mom_corrected = nusel_branches[mom_string]
        numu_muon_mom_corrected = ak.where(nusel_branches[contained_string] == 1, (numu_muon_mom_corrected - IntTrkMomRange) / GradTrkMomRange , numu_muon_mom_corrected) #contained
        numu_muon_mom_corrected = ak.where(nusel_branches[contained_string] == 0, (numu_muon_mom_corrected - IntTrkMomMCS) / GradTrkMomMCS , numu_muon_mom_corrected)    #uncontained
        numu_muon_mom_corrected = ak.where(nusel_branches[enu_string] < -990, -999.0, numu_muon_mom_corrected)
        numu_muon_corrected = ConvertMuonMomToEnergy(numu_muon_mom_corrected)
        # Correct hadron
        numu_had_corrected = nusel_branches[ehad_string]
        numu_had_corrected = ak.where(nusel_branches[contained_string] == 1, (numu_had_corrected - IntNuMuHadEnCont) / GradNuMuHadEnCont , numu_had_corrected)  #contained
        numu_had_corrected = ak.where(nusel_branches[contained_string] == 0, (numu_had_corrected - IntNuMuHadEnExit) / GradNuMuHadEnExit , numu_had_corrected) #uncontained
        numu_had_corrected = ak.where(nusel_branches[enu_string] < -990, -999.0, numu_had_corrected)
        # Add
        numu_corrected = numu_muon_corrected + numu_had_corrected
        numu_corrected = ak.where(nusel_branches[enu_string] < -990, -999.0, numu_corrected)

        nusel_branches = ak.with_field(
            nusel_branches,
            numu_corrected,
            f'Corrected{pid_string}NumuRecoE'
        )

    ##############################
    # Identify signal
    ##############################
    # CCnue
    signal_CC_nue_flav_mask = Signal.IsCCNueFlavourSignal(nusel_branches)
    outoffv_CC_nue_flav_mask = Signal.IsCCNueFlavourOutOfFV(nusel_branches)
    # CCnumu
    signal_CC_numu_flav_mask = Signal.IsCCNumuFlavourSignal(nusel_branches)
    outoffv_CC_numu_flav_mask = Signal.IsCCNumuFlavourOutOfFV(nusel_branches)
    # CCnutau
    CC_nutau_flav_mask = Signal.IsCCNutauFlavour(nusel_branches)
    # NC
    NC_mask = Signal.IsNC(nusel_branches)
    # Other
    other_mask = (~signal_CC_nue_flav_mask) & (~outoffv_CC_nue_flav_mask) & (~signal_CC_numu_flav_mask) & (~outoffv_CC_numu_flav_mask) & (~CC_nutau_flav_mask) & (~NC_mask)

    # Class masks (see Plots.py)
    class_masks = [signal_CC_nue_flav_mask, outoffv_CC_nue_flav_mask, signal_CC_numu_flav_mask, outoffv_CC_numu_flav_mask, CC_nutau_flav_mask, NC_mask, other_mask]

    ##############################
    # Plot signal
    ##############################
    signal_plot_dir = f'{args.plot_dir}/Signal/'

    fig, ax = plt.subplots(figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, signal_CC_numu_flav_mask, fig, ax, title='CCnumu Signal Events')
    Plots.save_plot(fig, f'{signal_plot_dir}/CCNumuSignal.pdf')
    
    fig, ax = plt.subplots(figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, signal_CC_nue_flav_mask, fig, ax, title='CCnue Signal Events')
    Plots.save_plot(fig, f'{signal_plot_dir}/CCNueSignal.pdf')

    ##############################
    # Plot CVN selection
    ##############################
    cvn_sel_CC_nue_mask = Selection.PassCCNueSelection_CVN(nusel_branches)
    cvn_sel_CC_numu_mask = Selection.PassCCNumuSelection_CVN(nusel_branches)

    cvn_plot_dir = f'{args.plot_dir}/CVNSelection/'
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, cvn_sel_CC_nue_mask, fig, ax[0], title='CVN Selected CCNue Spectrum')
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, cvn_sel_CC_numu_mask, fig, ax[1], title='CVN Selected CCNumu Spectrum')
    Plots.save_plot(fig, f'{cvn_plot_dir}/CVNSelection_TrueNuEnergy.pdf')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotSelectionMetrics(nusel_branches, signal_CC_nue_flav_mask, cvn_sel_CC_nue_mask, fig, ax[0], title='CVN CCnue selection metrics')
    Plots.PlotSelectionMetrics(nusel_branches, signal_CC_numu_flav_mask, cvn_sel_CC_numu_mask, fig, ax[1], title='CVN CCnumu selection metrics')
    Plots.save_plot(fig, f'{cvn_plot_dir}/CVNSelection_Metrics.pdf')

    #####################################
    # Plot pandizzle/pandrizzle selection
    #####################################
    izzle_sel_CC_nue_mask = Selection.PassCCNuePandrizzleSelection(nusel_branches)
    izzle_sel_CC_numu_mask = Selection.PassCCNumuPandizzleSelection(nusel_branches)

    izzle_plot_dir = f'{args.plot_dir}/IzzleSelection/'

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, izzle_sel_CC_nue_mask, fig, ax[0], title='Izzle Selected CCNue Spectrum')
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, izzle_sel_CC_numu_mask, fig, ax[1], title='Izzle Selected CCNumu Spectrum')
    Plots.save_plot(fig, f'{izzle_plot_dir}/IzzleSelection_TrueNuEnergy.pdf')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'CorrectedPandrizzleNueRecoE', class_masks, izzle_sel_CC_nue_mask, fig, ax[0], title='Izzle Selected CCNue Spectrum')
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'CorrectedPandizzleNumuRecoE', class_masks, izzle_sel_CC_numu_mask, fig, ax[1], title='Izzle Selected CCNumu Spectrum')
    Plots.save_plot(fig, f'{izzle_plot_dir}/IzzleSelection_RecoNuEnergy.pdf')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotSelectionMetrics(nusel_branches, signal_CC_nue_flav_mask, izzle_sel_CC_nue_mask, fig, ax[0], title='Izzle CCnue selection metrics')
    Plots.PlotSelectionMetrics(nusel_branches, signal_CC_numu_flav_mask, izzle_sel_CC_numu_mask, fig, ax[1], title='Izzle CCnumu selection metrics')
    Plots.save_plot(fig, f'{izzle_plot_dir}/IzzleSelection_Metrics.pdf')

    #####################################
    # Plot ivysaurus selection
    #####################################
    ivy_sel_CC_nue_mask = Selection.PassCCNueIvysaurusSelection(nusel_branches, 0.9)
    ivy_sel_CC_numu_mask = Selection.PassCCNumuIvysaurusSelection(nusel_branches)

    ivysaurus_plot_dir = f'{args.plot_dir}/IvysaurusSelection/'
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, ivy_sel_CC_nue_mask, fig, ax[0], title='Ivy Selected CCNue Spectrum')
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'Enu', class_masks, ivy_sel_CC_numu_mask, fig, ax[1], title='Ivy Selected CCNumu Spectrum')
    Plots.save_plot(fig, f'{ivysaurus_plot_dir}/IvysaurusSelection_TrueNuEnergy.pdf')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'CorrectedIvysaurusNueRecoE', class_masks, ivy_sel_CC_nue_mask, fig, ax[0], title='Ivy Selected CCNue Spectrum')
    Plots.PlotEnergySpectrumDecomposition(nusel_branches, 'CorrectedIvysaurusNumuRecoE', class_masks, ivy_sel_CC_numu_mask, fig, ax[1], title='Ivy Selected CCNumu Spectrum')
    Plots.save_plot(fig, f'{ivysaurus_plot_dir}/IvysaurusSelection_RecoNuEnergy.pdf')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6))
    Plots.PlotSelectionMetrics(nusel_branches, signal_CC_nue_flav_mask, ivy_sel_CC_nue_mask, fig, ax[0], title='Ivy CCnue selection metrics')
    Plots.PlotSelectionMetrics(nusel_branches, signal_CC_numu_flav_mask, ivy_sel_CC_numu_mask, fig, ax[1], title='Ivy CCnumu selection metrics')
    Plots.save_plot(fig, f'{ivysaurus_plot_dir}/IvysaurusSelection_Metrics.pdf')

##########################################################################################################
##########################################################################################################
    
def create_directory_structure(plot_dir) :
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    create_directory(plot_dir, 'Signal')
    create_directory(plot_dir, 'CVNSelection')
    create_directory(plot_dir, 'IzzleSelection')
    create_directory(plot_dir, 'IvysaurusSelection')

##########################################################################################################
##########################################################################################################
    
def create_directory(root_dir, dir_name) :
    if not os.path.isdir(f'{root_dir}/{dir_name}') :
        os.makedirs(f'{root_dir}/{dir_name}')

##########################################################################################################
##########################################################################################################
            
def parse_cli():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validation script for Pandora.")
    parser.add_argument("--plot_dir", type=str, required=True, help="Directory for storing plots.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--is_hd", type=str, required=True, help="If HD (True) or VD (False) file.")
    return parser.parse_args()

##########################################################################################################
##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    create_directory_structure(args.plot_dir)
    main(args)
