# Imports
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import os

import Plots

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
    nusel_branches = tree.arrays(['Run', 'SubRun', 'Event', 
                                  'RecoPFPTruePDG', 'RecoPFPTruePrimary', 'RecoPFPRecoCompleteness', 'RecoPFPRecoHitPurity',
                                  'RecoPFPIsPrimary', 'RecoPFPTrackShowerScore',
                                  'RecoTrackDeflecAngleSD', 'RecoTrackLength',
                                  'RecoTrackdEdxStart', 'RecoTrackdEdxEnd','RecoTrackdEdxEndRatio',
                                  'RecoTrackMichelNHits', 'RecoTrackMichelElectronMVA', 'RecoTrackMichelRecoEnergyPlane2',
                                  'RecoTrackEvalRatio', 'RecoTrackConcentration', 'RecoTrackCoreHaloRatio', 'RecoTrackConicalness',
                                  'RecoShowerPandrizzleDisplacement', 'RecoShowerPandrizzledEdxBestPlane', 'RecoShowerPandrizzleDCA',
                                  'RecoShowerPandrizzleWideness', 'RecoShowerPandrizzleEnergyDensity',
                                  'RecoShowerPandrizzleEvalRatio', 'RecoShowerPandrizzleConcentration', 'RecoShowerPandrizzleCoreHaloRatio', 'RecoShowerPandrizzleConicalness',
                                  'RecoShowerPandrizzleIsFilled',
                                  'RecoShowerPandrizzleMinLargestProjectedGapSize', 'RecoShowerPandrizzleMaxInitialGapSize',
                                  'RecoShowerPandrizzlePathwayLengthMin', 'RecoShowerPandrizzleMaxShowerStartPathwayScatteringAngle2D',
                                  'RecoShowerPandrizzleMaxNPostShowerStartHits', 'RecoShowerPandrizzleMaxPostShowerStartOpeningAngle',
                                  'RecoShowerPandrizzleMaxPostShowerStartShowerStartEnergyAsymmetry', 'RecoShowerPandrizzleMinPostShowerStartShowerStartMoliereRadius',
                                  'RecoShowerPandrizzleMaxFoundHitRatio',
                                  'RecoShowerPandrizzleMaxPostShowerStartScatterAngle', 'RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyWeightedMeanRadialDistance',
                                  'RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyAsymmetry',
                                  'RecoShowerPandrizzleNViewsWithAmbiguousHits', 'RecoShowerPandrizzleAmbiguousHitMaxUnaccountedEnergy',
                                  'RecoShowerPandrizzleModularPathwayLength',
                                  'RecoShowerPandrizzleModularNuVertexChargeWeightedMeanRadialDistance',
                                  'RecoShowerPandrizzleModularMaxNShowerHits'], library='ak')

    ##############
    # Global masks
    ##############
    completeness_mask = (nusel_branches['RecoPFPRecoCompleteness'] > args.completeness_thr)
    purity_mask = (nusel_branches['RecoPFPRecoHitPurity'] > args.purity_thr)
    hit_mask = (nusel_branches['RecoPFPRecoNHits'] > args.hit_thr)
    
    ############################################################
    # Pandizzle variables - Categories + binning match my thesis
    ############################################################
    pandizzle_vars_plot_dir = f'{args.plot_dir}/IzzleSelection/PandizzleBDTVars'
    track_length_PV = Plots.PlotVar('', 'Tracks', 'Track Length [cm]', [0, 1000], 30)
    wobbiliness_PV = Plots.PlotVar('', 'Tracks', 'Deviation From Straightness [cm]', [0, 40], 20)
    dedx_start_PV = Plots.PlotVar('', 'Tracks', 'Initial dE/dx [MeV/cm]', [0, 10], 20)
    dedx_end_PV = Plots.PlotVar('', 'Tracks', 'End dE/dx [MeV/cm]', [0, 10], 20)
    dedx_end_ratio_PV = Plots.PlotVar('', 'Tracks', 'End Region dE/dx Ratio', [0, 4], 20)
    michel_n_hits_PV =  Plots.PlotVar('', 'Tracks', 'Michel - Number of 2D Hits', [-2, 20], 22)
    michel_electron_mva_PV =  Plots.PlotVar('', 'Tracks', 'Michel - MVA Electron Score', [-2, 1], 12)
    michel_energy_PV =  Plots.PlotVar('', 'Tracks', 'Michel - Reco Energy [MeV]', [-2, 1], 12)
    track_eval_ratio_PV =  Plots.PlotVar('', 'Tracks', 'Eigenvalue Ratio', [0.0, 0.1], 20)
    track_concentration_PV =  Plots.PlotVar('', 'Tracks', 'Concentration [/cm]', [0, 20], 40)
    track_core_halo_PV =  Plots.PlotVar('', 'Tracks', 'Halo-Core Ratio', [0, 10], 20)
    track_conicalness_PV =  Plots.PlotVar('', 'Tracks', 'Conicalness [stupid units]', [0, 5], 20)

    # Target mask
    muon_target_mask = (nusel_branches['RecoPFPTruePrimary'] == 1) & (nusel_branches['RecoPFPIsPrimary'] == 1) & (nusel_branches['RecoPFPTrackShowerScore'] > 0.5) & completeness_mask & purity_mask & hit_mask

    # Apply to arrays
    true_pdg = ak.to_numpy(ak.flatten(nusel_branches['RecoPFPTruePDG'][muon_target_mask]))
    track_length = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackLength'][muon_target_mask]))
    deviation_from_straightness = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackDeflecAngleSD'][muon_target_mask]))
    dedx_start = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackdEdxStart'][muon_target_mask]))
    dedx_end = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackdEdxEnd'][muon_target_mask]))
    dedx_end_ratio = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackdEdxEndRatio'][muon_target_mask]))
    michel_n_hits = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackMichelNHits'][muon_target_mask]))
    michel_electron_mva = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackMichelElectronMVA'][muon_target_mask]))
    michel_energy = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackMichelRecoEnergyPlane2'][muon_target_mask]))
    track_eval_ratio = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackEvalRatio'][muon_target_mask]))
    track_concentration = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackConcentration'][muon_target_mask]))
    track_core_halo_ratio = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackCoreHaloRatio'][muon_target_mask]))
    track_conicalness = ak.to_numpy(ak.flatten(nusel_branches['RecoTrackConicalness'][muon_target_mask]))
    
    muon_signal_mask = (np.abs(true_pdg) == 13)

    # Topology vars
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
    Plots.PlotSignalBackgroundVar(track_length, muon_signal_mask, ~muon_signal_mask, track_length_PV, ax[0], x_label=track_length_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(deviation_from_straightness, muon_signal_mask, ~muon_signal_mask, wobbiliness_PV, ax[1], x_label=wobbiliness_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandizzle_vars_plot_dir}/TopologyVars.pdf')
    # Calorimetric vars
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 6))
    Plots.PlotSignalBackgroundVar(dedx_start, muon_signal_mask, ~muon_signal_mask, dedx_start_PV, ax[0], x_label=dedx_start_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(dedx_end, muon_signal_mask, ~muon_signal_mask, dedx_end_PV, ax[1], x_label=dedx_end_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(dedx_end_ratio, muon_signal_mask, ~muon_signal_mask, dedx_end_ratio_PV, ax[2], x_label=dedx_end_ratio_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandizzle_vars_plot_dir}/CalorimetryVars.pdf')
    # Michel vars
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 6))
    Plots.PlotSignalBackgroundVar(michel_n_hits, muon_signal_mask, ~muon_signal_mask, michel_n_hits_PV, ax[0], x_label=michel_n_hits_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(michel_electron_mva, muon_signal_mask, ~muon_signal_mask, michel_electron_mva_PV, ax[1], x_label=michel_electron_mva_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(michel_energy, muon_signal_mask, ~muon_signal_mask, michel_energy_PV, ax[2], x_label=michel_energy_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandizzle_vars_plot_dir}/MichelVars.pdf')
    # Track-shower vars (i.e. the warwick pid)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(track_concentration, muon_signal_mask, ~muon_signal_mask, track_concentration_PV, ax[0][0], x_label=track_concentration_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(track_core_halo_ratio, muon_signal_mask, ~muon_signal_mask, track_core_halo_PV, ax[0][1], x_label=track_core_halo_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(track_conicalness, muon_signal_mask, ~muon_signal_mask, track_conicalness_PV, ax[1][0], x_label=track_conicalness_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(track_eval_ratio, muon_signal_mask, ~muon_signal_mask, track_eval_ratio_PV, ax[1][1], x_label=track_eval_ratio_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandizzle_vars_plot_dir}/WarwickPIDVars.pdf')

    ############################################################
    # Pandrizzle variables - Categories + binning match my thesis
    ############################################################
    pandrizzle_vars_plot_dir = f'{args.plot_dir}/IzzleSelection/PandrizzleBDTVars'
    displacement_PV = Plots.PlotVar('', 'Showers', 'Displacement [cm]', [0, 99], 99)
    initial_dedx_PV = Plots.PlotVar('', 'Showers', 'Initial dE/dx [MeV/cm]', [-2, 15], 34)
    dca_PV = Plots.PlotVar('', 'Showers', 'Distance of Closest Approach [cm]', [0, 49], 49)
    wideness_PV = Plots.PlotVar('', 'Showers', 'Wideness [radians/cm]', [0.0, 0.03], 30)
    energy_density_PV = Plots.PlotVar('', 'Showers', 'Energy Density [MeV/cm3]', [-1.0, 1.5], 25)
    shower_eval_ratio_PV =  Plots.PlotVar('', 'Tracks', 'Eigenvalue Ratio', [0.0, 0.8], 20)
    shower_concentration_PV =  Plots.PlotVar('', 'Tracks', 'Concentration [/cm]', [0, 12], 24)
    shower_core_halo_PV =  Plots.PlotVar('', 'Tracks', 'Halo-Core Ratio', [0, 10], 20)
    shower_conicalness_PV =  Plots.PlotVar('', 'Tracks', 'Conicalness [stupid units]', [0, 20], 40)

    # Target mask
    electron_target_mask = (nusel_branches['RecoPFPTruePrimary'] == 1) & (nusel_branches['RecoPFPIsPrimary'] == 1) & (nusel_branches['RecoPFPTrackShowerScore'] < 0.5) & (nusel_branches['RecoPFPTrackShowerScore'] > 0.0) & completeness_mask & purity_mask & hit_mask

    # Apply to arrays
    true_pdg = ak.to_numpy(ak.flatten(nusel_branches['RecoPFPTruePDG'][electron_target_mask]))
    displacement = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleDisplacement'][electron_target_mask]))
    initial_dedx = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzledEdxBestPlane'][electron_target_mask]))
    dca = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzledEdxBestPlane'][electron_target_mask]))
    wideness = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleWideness'][electron_target_mask]))
    energy_density = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleEnergyDensity'][electron_target_mask]))
    shower_eval_ratio = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleEvalRatio'][electron_target_mask]))
    shower_concentration = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleConcentration'][electron_target_mask]))
    shower_core_halo_ratio = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleCoreHaloRatio'][electron_target_mask]))
    shower_conicalness = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleConicalness'][electron_target_mask]))

    electron_signal_mask = (np.abs(true_pdg) == 11)

    # Smoking gun vars
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
    Plots.PlotSignalBackgroundVar(displacement, electron_signal_mask, ~electron_signal_mask, displacement_PV, ax[0], x_label=displacement_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(initial_dedx, electron_signal_mask, ~electron_signal_mask, initial_dedx_PV, ax[1], x_label=initial_dedx_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(dca, electron_signal_mask, ~electron_signal_mask, dca_PV, ax[2], x_label=dca_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandrizzle_vars_plot_dir}/SmokingGunVars.pdf')
    # Others..
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 6))
    Plots.PlotSignalBackgroundVar(wideness, electron_signal_mask, ~electron_signal_mask, wideness_PV, ax[0], x_label=wideness_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(energy_density, electron_signal_mask, ~electron_signal_mask, energy_density_PV, ax[1], x_label=energy_density_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandrizzle_vars_plot_dir}/NotSmokingGunVars.pdf')
    # Track-shower vars (i.e. the warwick pid)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(shower_concentration, electron_signal_mask, ~electron_signal_mask, shower_concentration_PV, ax[0][0], x_label=shower_concentration_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(shower_core_halo_ratio, electron_signal_mask, ~electron_signal_mask, shower_core_halo_PV, ax[0][1], x_label=shower_core_halo_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(shower_conicalness, electron_signal_mask, ~electron_signal_mask, shower_conicalness_PV, ax[1][0], x_label=shower_conicalness_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(shower_eval_ratio, electron_signal_mask, ~electron_signal_mask, shower_eval_ratio_PV, ax[1][1], x_label=shower_eval_ratio_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{pandrizzle_vars_plot_dir}/WarwickPIDVars.pdf')

    ################################################################################
    # Pandrizzle Connection Pathway Variables - Categories + binning match my thesis
    ################################################################################
    enhanced_pandrizzle_vars_plot_dir = f'{args.plot_dir}/IzzleSelection/EnhancedPandrizzleBDTVars'
    largest_gap_PV = Plots.PlotVar('', 'Showers', 'Initial Region Gap Size [cm]', [-1.0, 2.0], 30)
    initial_gap_PV = Plots.PlotVar('', 'Showers', 'Initial Gap Size [cm]', [-2.0, 4.0], 60)
    pathway_length_PV = Plots.PlotVar('', 'Showers', 'Pathway Length [cm]', [-2.0, 30.0], 32)
    pathway_deviation_PV = Plots.PlotVar('', 'Showers', 'Connection Pathway Deviation [cm]', [-2.0, 10.0], 24)
    shower_n_hits_PV = Plots.PlotVar('', 'Showers', 'Number of Shower Hits', [-2.0, 2000.0], 75)
    shower_opening_angle_PV = Plots.PlotVar('', 'Showers', 'Opening Angle', [-1.0, 20.0], 42)
    shower_energy_asymmetry_PV = Plots.PlotVar('', 'Showers', 'Shower Energy Asymmetry', [-0.4, 1.0], 28)
    moliere_PV = Plots.PlotVar('', 'Showers', 'Moliere Radius', [-2, 10.0], 48)
    found_hit_ratio_PV = Plots.PlotVar('', 'Showers', 'Found Hit Ratio', [-0.4, 1.5], 19)
    scatter_angle_PV = Plots.PlotVar('', 'Showers', 'Scatter Angle [degrees]', [-2.0, 30.0], 32)
    cp_energy_asymmetry_PV = Plots.PlotVar('', 'Showers', 'Connection Pathway Energy Asymmetry', [-0.4, 1.0], 28)
    cp_energy_weighted_mean_radial_dist_PV = Plots.PlotVar('', 'Showers', 'Connection Pathway Energy Weighted Mean Radial Distance [cm]', [-2.0, 20.0], 44)
    n_amb_views_PV = Plots.PlotVar('', 'Showers', 'Number of Ambiguous Hit Views', [-2.0, 4.0], 60)
    amb_hit_energy_PV = Plots.PlotVar('', 'Showers', 'Ambiguous Hit Unaccounted Energy [MeV]', [-10.0, 5.0], 30)

    cp_target_mask = (nusel_branches['RecoPFPTruePrimary'] == 1) & (nusel_branches['RecoPFPIsPrimary'] == 1) &\
        (nusel_branches['RecoPFPTrackShowerScore'] < 0.5) & (nusel_branches['RecoShowerPandrizzlePathwayLengthMin'] > -990) & (nusel_branches['RecoPFPTrackShowerScore'] > 0.0) & completeness_mask & purity_mask & hit_mask

    # Apply to arrays
    true_pdg = ak.to_numpy(ak.flatten(nusel_branches['RecoPFPTruePDG'][cp_target_mask]))
    largest_gap = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMinLargestProjectedGapSize'][cp_target_mask]))
    initial_gap = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxInitialGapSize'][cp_target_mask]))
    pathway_length = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzlePathwayLengthMin'][cp_target_mask]))
    pathway_deviation = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxShowerStartPathwayScatteringAngle2D'][cp_target_mask]))
    shower_n_hits = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxNPostShowerStartHits'][cp_target_mask]))
    shower_opening_angle = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxPostShowerStartOpeningAngle'][cp_target_mask]))
    shower_energy_asymmetry = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxPostShowerStartShowerStartEnergyAsymmetry'][cp_target_mask]))
    moliere = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMinPostShowerStartShowerStartMoliereRadius'][cp_target_mask]))
    found_hit_ratio = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxFoundHitRatio'][cp_target_mask]))
    scatter_angle = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxPostShowerStartScatterAngle'][cp_target_mask]))
    cp_energy_asymmetry = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyAsymmetry'][cp_target_mask]))
    cp_energy_weighted_mean_radial_dist = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleMaxPostShowerStartNuVertexEnergyWeightedMeanRadialDistance'][cp_target_mask]))
    n_amb_views = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleNViewsWithAmbiguousHits'][cp_target_mask]))
    amb_hit_energy = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleAmbiguousHitMaxUnaccountedEnergy'][cp_target_mask]))

    electron_signal_mask = (np.abs(true_pdg) == 11)

    # Inital region vars
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(largest_gap, electron_signal_mask, ~electron_signal_mask, largest_gap_PV, ax[0], x_label=largest_gap_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(initial_gap, electron_signal_mask, ~electron_signal_mask, initial_gap_PV, ax[1], x_label=initial_gap_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{enhanced_pandrizzle_vars_plot_dir}/InitialRegionVars.pdf')
    # Connection pathway vars
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(pathway_length, electron_signal_mask, ~electron_signal_mask, pathway_length_PV, ax[0], x_label=pathway_length_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(pathway_deviation, electron_signal_mask, ~electron_signal_mask, pathway_deviation_PV, ax[1], x_label=pathway_deviation_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{enhanced_pandrizzle_vars_plot_dir}/ConnectionPathwayVars.pdf')
    # Shower region variables - does it look like a shower?
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(shower_n_hits, electron_signal_mask, ~electron_signal_mask, shower_n_hits_PV, ax[0][0], x_label=shower_n_hits_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(shower_opening_angle, electron_signal_mask, ~electron_signal_mask, shower_opening_angle_PV, ax[0][1], x_label=shower_opening_angle_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(shower_energy_asymmetry, electron_signal_mask, ~electron_signal_mask, shower_energy_asymmetry_PV, ax[1][0], x_label=shower_energy_asymmetry_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(moliere, electron_signal_mask, ~electron_signal_mask, moliere_PV, ax[1][1], x_label=moliere_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(found_hit_ratio, electron_signal_mask, ~electron_signal_mask, found_hit_ratio_PV, ax[2][0], x_label=found_hit_ratio_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{enhanced_pandrizzle_vars_plot_dir}/ShowerRegionVars.pdf')
    # Shower region variables - do the connection pathway and shower region align?
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(scatter_angle, electron_signal_mask, ~electron_signal_mask, scatter_angle_PV, ax[0], x_label=scatter_angle_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(cp_energy_asymmetry, electron_signal_mask, ~electron_signal_mask, cp_energy_asymmetry_PV, ax[1], x_label=cp_energy_asymmetry_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(cp_energy_weighted_mean_radial_dist, electron_signal_mask, ~electron_signal_mask, cp_energy_weighted_mean_radial_dist_PV, ax[2], x_label=cp_energy_weighted_mean_radial_dist_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{enhanced_pandrizzle_vars_plot_dir}/AlignmentVars.pdf')
    # Ambiguous
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(n_amb_views, electron_signal_mask, ~electron_signal_mask, n_amb_views_PV, ax[0], x_label=n_amb_views_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(amb_hit_energy, electron_signal_mask, ~electron_signal_mask, amb_hit_energy_PV, ax[1], x_label=amb_hit_energy_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{enhanced_pandrizzle_vars_plot_dir}/AmbiguousVars.pdf')

    #######################################################################################
    # Backup Pandrizzle Connection Pathway Variables - Categories + binning match my thesis
    #######################################################################################
    backup_pandrizzle_vars_plot_dir = f'{args.plot_dir}/IzzleSelection/BackupPandrizzleBDTVars'
    modular_pathway_length_PV = Plots.PlotVar('', 'Showers', 'Modular - Pathway Length [cm]', [-2.0, 30.0], 32)
    modular_cp_energy_weighted_mean_radial_dist_PV = Plots.PlotVar('', 'Showers', 'Modular - Connection Pathway Energy Weighted Mean Radial Distance [cm]', [-2.0, 20.0], 44)
    modular_shower_n_hits_PV = Plots.PlotVar('', 'Showers', 'Modular - Number of Shower Hits', [-2.0, 2000.0], 75)

    modular_target_mask = (nusel_branches['RecoPFPTruePrimary'] == 1) & (nusel_branches['RecoPFPIsPrimary'] == 1) &\
        (nusel_branches['RecoPFPTrackShowerScore'] < 0.5) & (nusel_branches['RecoShowerPandrizzleModularPathwayLength'] > -990) &\
        (nusel_branches['RecoPFPTrackShowerScore'] > 0.0) & completeness_mask & purity_mask & hit_mask

    # Apply to arrays
    true_pdg = ak.to_numpy(ak.flatten(nusel_branches['RecoPFPTruePDG'][modular_target_mask]))
    modular_pathway_length = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleModularPathwayLength'][modular_target_mask]))
    modular_cp_energy_weighted_mean_radial_dist = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleModularNuVertexChargeWeightedMeanRadialDistance'][modular_target_mask]))
    modular_shower_n_hits = ak.to_numpy(ak.flatten(nusel_branches['RecoShowerPandrizzleModularMaxNShowerHits'][modular_target_mask]))

    electron_signal_mask = (np.abs(true_pdg) == 11)

    # Modular shower vars
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 10))
    Plots.PlotSignalBackgroundVar(modular_pathway_length, electron_signal_mask, ~electron_signal_mask, modular_pathway_length_PV, ax[0], x_label=modular_pathway_length_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(modular_cp_energy_weighted_mean_radial_dist, electron_signal_mask, ~electron_signal_mask, modular_cp_energy_weighted_mean_radial_dist_PV, ax[1], x_label=modular_cp_energy_weighted_mean_radial_dist_PV.x_label, title='', show_under_over_flow=True)
    Plots.PlotSignalBackgroundVar(modular_shower_n_hits, electron_signal_mask, ~electron_signal_mask, modular_shower_n_hits_PV, ax[2], x_label=modular_shower_n_hits_PV.x_label, title='', show_under_over_flow=True)
    Plots.save_plot(fig, f'{backup_pandrizzle_vars_plot_dir}/ModularShowerVars.pdf')

##########################################################################################################
##########################################################################################################
    
def create_directory_structure(plot_dir) :
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    create_directory(plot_dir, 'IzzleSelection')
    create_directory(f'{plot_dir}/IzzleSelection', 'PandizzleBDTVars')
    create_directory(f'{plot_dir}/IzzleSelection', 'PandrizzleBDTVars')
    create_directory(f'{plot_dir}/IzzleSelection', 'EnhancedPandrizzleBDTVars')
    create_directory(f'{plot_dir}/IzzleSelection', 'BackupPandrizzleBDTVars')

##########################################################################################################
##########################################################################################################
    
def create_directory(root_dir, dir_name) :
    if not os.path.isdir(f'{root_dir}/{dir_name}') :
        os.makedirs(f'{root_dir}/{dir_name}')

##########################################################################################################
##########################################################################################################
            
def parse_cli():
    parser = argparse.ArgumentParser(description="Validation script for Pandora.")
    parser.add_argument("--plot_dir", type=str, required=True, help="Directory for storing plots.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--completeness_thr", type=float, required=False, default=0.0, help="Completeness threshold.")
    parser.add_argument("--purity_thr", type=float, required=False, default=0.0, help="Purity threshold.")
    parser.add_argument("--hit_thr", type=float, required=False, default=0.0, help="NHit2D threshold.")    
    return parser.parse_args()

##########################################################################################################
##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    create_directory_structure(args.plot_dir)
    main(args)
