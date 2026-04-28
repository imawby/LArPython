# Imports
import argparse
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import sklearn 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import class_weight
import os

import Plots

##########################################################################################################
##########################################################################################################

class PIDMethod :
    def __init__(self, dir_name, pid_name, n_classes, branch_names, signal_pdgs, input_type_name, range, n_bins):
        self.dir_name = dir_name
        self.pid_name = pid_name
        self.n_classes = n_classes
        self.branch_names = branch_names
        self.signal_pdgs = signal_pdgs
        self.input_type_name = input_type_name
        self.n_bins = n_bins
        self.range = range

##########################################################################################################
##########################################################################################################

def PlotROC(fig, ax, pid_var, signal_masks, pid_scores) :
    falsePositive = dict()
    bkgRejection = dict()
    truePositive = dict()
    roc = dict()

    for i_type in range(pid_var.n_classes):
        falsePositive[i_type], truePositive[i_type], _ = roc_curve(signal_masks[i_type].astype(int), pid_scores[i_type])
        bkgRejection[i_type] = 1 - falsePositive[i_type]
        roc[i_type] = sklearn.metrics.auc(falsePositive[i_type], bkgRejection[i_type])

    # Plot
    rocCurveTitles = pid_var.signal_pdgs
    for i_type in range(pid_var.n_classes):
        ax[i_type].plot(truePositive[i_type], bkgRejection[i_type], label=f'ROC curve (area = {roc[i_type]:0.2f})')
        ax[i_type].plot([0, 1], [0, 1], 'k--')
        ax[i_type].set_xlim(0.0, 1.0)
        ax[i_type].set_ylim(0.0, 1.0)
        ax[i_type].set_xticks(np.arange(0, 1.1, 0.1))
        ax[i_type].set_yticks(np.arange(0, 1.1, 0.1))
        ax[i_type].set_xlabel("Efficiency")
        ax[i_type].set_ylabel("BG Rejection")
        ax[i_type].set_title(f"PDG={rocCurveTitles[i_type]}")
        ax[i_type].legend(loc="lower right")
        ax[i_type].grid(True)

##########################################################################################################
##########################################################################################################

def PlotConfusionMatrix(fig, ax, pid_var, signal_masks, pid_scores):

    y_true = np.vstack(signal_masks)
    y_pred = np.vstack(pid_scores)

    confMatrix = confusion_matrix(y_true.argmax(axis=0), y_pred.argmax(axis=0))

    trueSums = np.sum(confMatrix, axis=1)
    predSums = np.sum(confMatrix, axis=0)

    trueNormalised = np.zeros(shape=(pid_var.n_classes, pid_var.n_classes))
    predNormalised = np.zeros(shape=(pid_var.n_classes, pid_var.n_classes))

    for trueIndex in range(pid_var.n_classes):
        for predIndex in range(pid_var.n_classes):
            nEntries = confMatrix[trueIndex][predIndex]

            if trueSums[trueIndex] > 0 :
                trueNormalised[trueIndex][predIndex] = (
                    float(nEntries) / float(trueSums[trueIndex])
                )

            if predSums[predIndex] > 0 :
                predNormalised[trueIndex][predIndex] = (
                    float(nEntries) / float(predSums[predIndex])
                )

    displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=pid_var.signal_pdgs)
    displayTrueNorm.plot(ax=ax[0], cmap='Blues', colorbar=False)

    displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=pid_var.signal_pdgs)
    displayPredNorm.plot(ax=ax[1], cmap='Blues', colorbar=False)        
            
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
    nusel_branches = tree.arrays(['Run', 'SubRun', 'Event', 'RecoPFPTruePrimary',
                                  'NuPdg', 'NC',
                                  'NuX', 'NuY', 'NuZ', 'TargetZ',
                                  'RecoPFPIsPrimary', 'RecoPFPTrackShowerScore', 'RecoPFPRecoNHits',
                                  'RecoPFPRecoCompleteness', 'RecoPFPRecoHitPurity',
                                  'RecoPFPTruePDG', 'RecoTrackPandizzleVar',
                                  'IvysaurusMuonScore', 'IvysaurusProtonScore', 'IvysaurusPionScore', 'IvysaurusElectronScore', 'IvysaurusPhotonScore',
                                  'RecoShowerEnhancedPandrizzleScore', 'RecoShowerBackupPandrizzleScore'], library='ak')

    #########################
    # Create PID objects
    #########################
    pandizzle_pid = PIDMethod('IzzleSelection', 'Pandizzle', 1, ['RecoTrackPandizzleVar'], [13], 'Tracks', [-1.0, 1.0], 40)
    enhanced_pandrizzle_pid = PIDMethod('IzzleSelection', 'EnhancedPandrizzle', 1, ['RecoShowerEnhancedPandrizzleScore'], [11], 'Showers', [-1.0, 1.0], 40)
    backup_pandrizzle_pid = PIDMethod('IzzleSelection', 'BackupPandizzle', 1, ['RecoShowerBackupPandrizzleScore'], [11], 'Showers', [-1.0, 1.0], 40)
    ivysaurus_pid = PIDMethod('IvysaurusSelection', 'Ivysaurus', 5, ['IvysaurusMuonScore', 'IvysaurusProtonScore', 'IvysaurusPionScore', 'IvysaurusElectronScore', 'IvysaurusPhotonScore'], [13, 2212, 211, 11, 22], 'Particles', [-1.0, 1.0], 40)

    for pid_var in [pandizzle_pid, enhanced_pandrizzle_pid, backup_pandrizzle_pid, ivysaurus_pid] :

        #########################
        # Define target
        #########################
        track_shower_mask = (nusel_branches['RecoPFPTrackShowerScore'] > 0.5) if pid_var.input_type_name == 'Tracks' else ((nusel_branches['RecoPFPTrackShowerScore'] > 0.0) & (nusel_branches['RecoPFPTrackShowerScore'] < 0.5)) if pid_var.input_type_name == 'Showers' else ak.ones_like(nusel_branches['RecoPFPTrackShowerScore'], dtype=bool)
        completeness_mask = (nusel_branches['RecoPFPRecoCompleteness'] > args.completeness_thr)
        purity_mask = (nusel_branches['RecoPFPRecoHitPurity'] > args.purity_thr)
        hit_mask = (nusel_branches['RecoPFPRecoNHits'] > args.hit_thr)
        target_mask = (nusel_branches['RecoPFPTruePrimary'] == 1) & (nusel_branches['RecoPFPIsPrimary'] == 1) & track_shower_mask & completeness_mask & purity_mask & hit_mask

        # Get plotting arrays
        true_pdg = ak.to_numpy(ak.flatten(nusel_branches['RecoPFPTruePDG'][target_mask]))
        pid_scores = []
        signal_pdgs = []
        signal_masks = []

        for i_type in range(pid_var.n_classes) :
            pid_scores.append(ak.to_numpy(ak.flatten(nusel_branches[pid_var.branch_names[i_type]][target_mask])))
            signal_masks.append((np.abs(true_pdg) == pid_var.signal_pdgs[i_type]))
            signal_pdgs.append(pid_var.signal_pdgs[i_type])


        ############################
        # Plot classification scores
        ############################
        fig, ax = plt.subplots(ncols=1, nrows=pid_var.n_classes, figsize=((10, 6) if pid_var.n_classes == 1 else (10, 26)))        
        if (type(ax) != np.ndarray) :
            ax = [ax]

        PlotROC(fig, ax, pid_var, signal_masks, pid_scores)
        Plots.save_plot(fig, f'{args.plot_dir}/{pid_var.dir_name}/PIDPerformance/{pid_var.pid_name}_ROC.pdf')

        ############################
        # Plot confusion
        ############################
        if (pid_var.n_classes > 1) :
            fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10, 12))
            PlotConfusionMatrix(fig, ax, pid_var, signal_masks, pid_scores)
            Plots.save_plot(fig, f'{args.plot_dir}/{pid_var.dir_name}/PIDPerformance/{pid_var.pid_name}_Confusion.pdf')
        
##########################################################################################################
##########################################################################################################
    
def create_directory_structure(plot_dir) :
    if not os.path.isdir(plot_dir) :
        os.makedirs(plot_dir)

    create_directory(plot_dir, 'IzzleSelection')
    create_directory(f'{plot_dir}/IzzleSelection', 'PIDPerformance')
    create_directory(f'{plot_dir}/IvysaurusSelection', 'PIDPerformance')
    
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
