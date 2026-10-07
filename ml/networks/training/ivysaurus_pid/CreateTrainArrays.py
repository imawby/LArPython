import argparse
import numpy as np
import math

import FileHelper
import MapKeys

########################################################################################################

def split_array(arr, train_indices, test_indices):
    return arr[train_indices], arr[test_indices]

########################################################################################################

def main(args):
    # start/end grid: args.dimensions x args.dimensions grid of energy deposits in the start/end region
    # start/end grid valid: args.dimensions x args.dimensions grid of 1s and 0s indicating whether the corresponding start/end grid cell is filled
    # trackVars:  nTrackChildren, nTrackChildren_valid, nShowerChildren, nShowerChildren_valid, nGrandChildren, nGrandChildren_valid,
    #             nChildHits, nChildHits_valid, childEnergy, childEnergy_valid, childTrackScore, childTrackScore_valid,
    #             trackLength, trackLength_valid, trackWobble, trackWobble_valid, momComparison, momComparison_valid, nHits2D, trackScore,
    #             distToEdge_norm, isPrimary
    # showerVars: displacement, displacement_valid, dca, dca_valid, trackStubLength, trackStubLength_valid,
    #             nuVertexAvSeparation, nuVertexAvSeparation_valid, nuVertexChargeAsymmetry, nuVertexChargeAsymmetry_valid
    (gridMap, trackVars, showerVars,
     y, distToEdge, isPrimary) = FileHelper.readTree(args)

    # Split into test/train and save
    nEntries = y.shape[0]
    ntest = math.floor(nEntries * 0.1)
    ntrain = math.floor(nEntries * 0.9)              
    print('n_test:', ntest)
    print('n_train:', ntrain)
    indices = np.arange(nEntries)
    np.random.shuffle(indices)
    train_idx = indices[:ntrain]
    test_idx = indices[ntrain:ntrain + ntest]
    # Grids
    gridMap_train = {key: gridMap[key][train_idx] for key in MapKeys.GRID_KEYS}
    gridMap_test = {key: gridMap[key][test_idx] for key in MapKeys.GRID_KEYS}
    # Other variables
    trackVars_train, trackVars_test = split_array(trackVars, train_idx, test_idx)
    showerVars_train, showerVars_test = split_array(showerVars, train_idx, test_idx)
    y_train, y_test = split_array(y, train_idx, test_idx)
    distToEdge_train, distToEdge_test = split_array(distToEdge, train_idx, test_idx)
    isPrimary_train, isPrimary_test = split_array(isPrimary, train_idx, test_idx)

    # Print shapes
    print('--------------------------------------')
    for key in MapKeys.GRID_KEYS:
        print(f'{key} (train, test): {gridMap_train[key].shape}, {gridMap_test[key].shape}')
    print(f'trackVars (train, test): {trackVars_train.shape}, {trackVars_test.shape}')
    print(f'showerVars (train, test): {showerVars_train.shape}, {showerVars_test.shape}')
    print(f'y (train, test): {y_train.shape}, {y_test.shape}')

    class_labels = ['Muon', 'Proton', 'Pion', 'Electron', 'lowEElectron','Photon']
    particleType_train = np.argmax(y_train, axis=1)
    class_counts_train = np.bincount(particleType_train, minlength=len(class_labels))
    particleType_test = np.argmax(y_test, axis=1)
    class_counts_test = np.bincount(particleType_test, minlength=len(class_labels))
    
    for i in range(len(class_labels)):
        print(f'n_{class_labels[i]} (train, test): {class_counts_train[i]}, {class_counts_test[i]}')
    print('--------------------------------------')

    # Save file
    output_file_name = f"{args.output_dir}/{args.file_name.replace('.root', '')}.npz"        
    np.savez(output_file_name,
             startU_train=gridMap_train['startU'], startU_mask_train=gridMap_train['startU_mask'],
             startV_train=gridMap_train['startV'], startV_mask_train=gridMap_train['startV_mask'],
             startW_train=gridMap_train['startW'], startW_mask_train=gridMap_train['startW_mask'],
             startU_test=gridMap_test['startU'], startU_mask_test=gridMap_test['startU_mask'],
             startV_test=gridMap_test['startV'], startV_mask_test=gridMap_test['startV_mask'],
             startW_test=gridMap_test['startW'], startW_mask_test=gridMap_test['startW_mask'],
             endU_train=gridMap_train['endU'], endU_mask_train=gridMap_train['endU_mask'],
             endV_train=gridMap_train['endV'], endV_mask_train=gridMap_train['endV_mask'],
             endW_train=gridMap_train['endW'], endW_mask_train=gridMap_train['endW_mask'],
             endU_test=gridMap_test['endU'], endU_mask_test=gridMap_test['endU_mask'],
             endV_test=gridMap_test['endV'], endV_mask_test=gridMap_test['endV_mask'],
             endW_test=gridMap_test['endW'], endW_mask_test=gridMap_test['endW_mask'],
             trackVars_train=trackVars_train,
             trackVars_test=trackVars_test,
             showerVars_test=showerVars_test,
             showerVars_train=showerVars_train,
             y_train=y_train,
             y_test=y_test,
             distToEdge_train=distToEdge_train,
             distToEdge_test=distToEdge_test,
             isPrimary_train=isPrimary_train,
             isPrimary_test=isPrimary_test)                 

########################################################################################################        

def parse_cli():
    parser = argparse.ArgumentParser(description="Ivysaurus PID")

    parser.add_argument("--file_name", type=str, required=True, help="Name of file to process")
    parser.add_argument("--input_dir", type=str, required=True, help="Input file directory")    
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save processed files")
    parser.add_argument("--pdgs", type=int, nargs="+", default=[13, 2212, 211, 11, 22], help="PDG IDs to collect - ATTN: order matters!")
    parser.add_argument("--detector", type=str, default="dune_hd", help="Detector: dune_hd, ")
    parser.add_argument("--dimensions", type=int, default=24, help="Grid dimensions")
    parser.add_argument("--is_contained", action="store_true", help="Training for contained particles?")

    return parser.parse_args()

########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    main(args)

