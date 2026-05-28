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
    
def ConvertMuonMomToEnergy(muon_mom_array) :
    muon_mass = 0.1056583745
    jam = np.sqrt((muon_mom_array * muon_mom_array) + (muon_mass * muon_mass))
    jam = ak.where(muon_mom_array < -990, -999.0, jam)
    return jam

##########################################################################################################

def main(args) :

    # Detector
    detector_config = Definitions.DetectorConfig(Definitions.dune_hd_fd_energy_corr, Definitions.dune_hd_fd_fv, Definitions.dune_hd_fd_cvn, Definitions.dune_hd_fd_izzle, Definitions.dune_hd_fd_ivysaurus) if args.detector == "hd_fd" else None

    if detector_config is None:
        raise NotImplementedError("other detector types not configured")
    
    file_name = f'{args.input_file}'
    with uproot.open(file_name) as file:
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

    ##########################
    # Correct reco nue energy
    ##########################
    IntShwEnergy = detector_config.energy_corr["IntShwEnergy"]
    GradShwEnergy = detector_config.energy_corr["GradShwEnergy"]
    IntNuEHadEn = detector_config.energy_corr["IntNuEHadEn"]
    GradNuEHadEn = detector_config.energy_corr["GradNuEHadEn"]

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

    ##############################
    # Correct reco numu energy
    ##############################        
    IntTrkMomRange = detector_config.energy_corr["IntTrkMomRange"]
    GradTrkMomRange = detector_config.energy_corr["GradTrkMomRange"]
    IntTrkMomMCS = detector_config.energy_corr["IntTrkMomMCS"]
    GradTrkMomMCS = detector_config.energy_corr["GradTrkMomMCS"]
    IntNuMuHadEnCont = detector_config.energy_corr["IntNuMuHadEnCont"]
    GradNuMuHadEnCont = detector_config.energy_corr["GradNuMuHadEnCont"]
    IntNuMuHadEnExit = detector_config.energy_corr["IntNuMuHadEnExit"]
    GradNuMuHadEnExit = detector_config.energy_corr["GradNuMuHadEnExit"]

    for pid_string in ['Pandizzle', 'Ivysaurus'] :
        enu_string = f'Sel{pid_string}TrackNumuEnu'
        mom_string = f'Sel{pid_string}TrackRecoMom'
        contained_string = f'Sel{pid_string}TrackContained'
        ehad_string = f'Sel{pid_string}TrackNumuEHad'

        # Correct muon
        numu_muon_mom_corrected = nusel_branches[mom_string]
        numu_muon_mom_corrected = ak.where(nusel_branches[contained_string] == 1, (numu_muon_mom_corrected - IntTrkMomRange) / GradTrkMomRange , numu_muon_mom_corrected) #contained
        numu_muon_mom_corrected = ak.where(nusel_branches[contained_string] == 0, (numu_muon_mom_corrected - IntTrkMomMCS) / GradTrkMomMCS , numu_muon_mom_corrected)     #uncontained
        numu_muon_mom_corrected = ak.where(nusel_branches[enu_string] < 0.0, -999.0, numu_muon_mom_corrected)
        numu_muon_corrected = ConvertMuonMomToEnergy(numu_muon_mom_corrected)
        # Correct hadron
        numu_had_corrected = nusel_branches[ehad_string]
        numu_had_corrected = ak.where(nusel_branches[contained_string] == 1, (numu_had_corrected - IntNuMuHadEnCont) / GradNuMuHadEnCont , numu_had_corrected)  #contained
        numu_had_corrected = ak.where(nusel_branches[contained_string] == 0, (numu_had_corrected - IntNuMuHadEnExit) / GradNuMuHadEnExit , numu_had_corrected)  #uncontained
        numu_had_corrected = ak.where(nusel_branches[enu_string] < 0.0, -999.0, numu_had_corrected)
        # Add
        numu_corrected = numu_muon_corrected + numu_had_corrected
        numu_corrected = ak.where(nusel_branches[enu_string] < 0.0, -999.0, numu_corrected)

        nusel_branches = ak.with_field(
            nusel_branches,
            numu_corrected,
            f'Corrected{pid_string}NumuRecoE'
        )

    ##############################
    # Identify signal
    ##############################
    # CCnue
    signal_CC_nue_flav_mask = Signal.IsCCNueFlavourSignal(nusel_branches, detector_config.fv)
    outoffv_CC_nue_flav_mask = Signal.IsCCNueFlavourOutOfFV(nusel_branches, detector_config.fv)
    # CCnumu
    signal_CC_numu_flav_mask = Signal.IsCCNumuFlavourSignal(nusel_branches, detector_config.fv)
    outoffv_CC_numu_flav_mask = Signal.IsCCNumuFlavourOutOfFV(nusel_branches, detector_config.fv)
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
    cvn_sel_CC_nue_mask = Selection.PassCCNueSelection_CVN(nusel_branches, detector_config)
    cvn_sel_CC_numu_mask = Selection.PassCCNumuSelection_CVN(nusel_branches, detector_config)

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
    izzle_sel_CC_nue_mask = Selection.PassCCNueSelection_Izzle(nusel_branches, detector_config)
    izzle_sel_CC_numu_mask = Selection.PassCCNumuSelection_Izzle(nusel_branches, detector_config)

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
    ivy_sel_CC_nue_mask = Selection.PassCCNueSelection_Ivysaurus(nusel_branches, detector_config, 0.9)
    ivy_sel_CC_numu_mask = Selection.PassCCNumuSelection_Ivysaurus(nusel_branches, detector_config)

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
    
def create_directory_structure(plot_dir) :
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    create_directory(plot_dir, 'Signal')
    create_directory(plot_dir, 'CVNSelection')
    create_directory(plot_dir, 'IzzleSelection')
    create_directory(plot_dir, 'IvysaurusSelection')

##########################################################################################################
    
def create_directory(root_dir, dir_name) :
    if not os.path.isdir(f'{root_dir}/{dir_name}') :
        os.makedirs(f'{root_dir}/{dir_name}')

##########################################################################################################
            
def parse_cli():
    parser = argparse.ArgumentParser(description="Validation script for Pandora.")
    parser.add_argument("--plot_dir", type=str, required=True, help="Directory for storing plots.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--detector", type=str, required=True, help="hd_fd, hd_vd, sbnd, icarus")
    return parser.parse_args()

##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    create_directory_structure(args.plot_dir)
    main(args)

