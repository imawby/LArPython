import argparse
import numpy as np
import math

import FileHelper

########################################################################################################

detector_boundaries = {
"dune_hd" : {
        "MinX":(-360.0 + 10.0),
        "MaxX":(360.0 - 10.0),
        "MinY":(-600.0 + 10.0),
        "MaxY":(600.0 - 10.0),
        "MinZ":(0 + 10.0),
        "MaxZ":(1394.0 - 10.0)
    }
}

########################################################################################################

def main(args):
    # Split training sample into contained and exiting
    this_detector_boundaries = detector_boundaries.get(args.detector)

    (startGridU, startGridU_valid,
     startGridV, startGridV_valid,
     startGridW, startGridW_valid,
     endGridU, endGridU_valid,
     endGridV, endGridV_valid,
     endGridW, endGridW_valid,
     pfpVars, trackVars, showerVars,
     y) = FileHelper.readTree(args, this_detector_boundaries)
    
    nEntries = startGridU.shape[0]

    if this_detector_boundaries is None:
        raise ValueError(f"Unknown detector type {args.detector}. Available options: {list(detector_boundaries.keys())}")
    
    contained_mask = (pfpVars[:,0] > this_detector_boundaries["MinX"]) & (pfpVars[:,0] < this_detector_boundaries["MaxX"]) &\
        (pfpVars[:,1] > this_detector_boundaries["MinY"]) & (pfpVars[:,1] < this_detector_boundaries["MaxY"]) &\
        (pfpVars[:,2] > this_detector_boundaries["MinZ"]) & (pfpVars[:,2] < this_detector_boundaries["MaxZ"])
    
    print('n_contained:', np.sum(contained_mask))
    print('n_exiting:', np.sum(~contained_mask))

    for is_contained in [True, False] :        
        target_mask = (contained_mask == is_contained)
        n_entries = np.sum(target_mask)
        ntest = math.floor(n_entries * 0.1)
        ntrain = math.floor(n_entries * 0.9)
        
        print('Contained' if is_contained else 'Exiting')        
        print('n_test:', ntest)
        print('n_train:', ntrain)

        indices = np.flatnonzero(target_mask)
        np.random.shuffle(indices)
        train_idx = indices[:ntrain]
        test_idx = indices[ntrain:ntrain + ntest]

        startGridU_train = startGridU[train_idx]
        startGridV_train = startGridV[train_idx]
        startGridW_train = startGridW[train_idx]
        startGridU_test = startGridU[test_idx]
        startGridV_test = startGridV[test_idx]
        startGridW_test = startGridW[test_idx]
        startGridU_valid_train = startGridU_valid[train_idx]
        startGridV_valid_train = startGridV_valid[train_idx]
        startGridW_valid_train = startGridW_valid[train_idx]
        startGridU_valid_test = startGridU_valid[test_idx]
        startGridV_valid_test = startGridV_valid[test_idx]
        startGridW_valid_test = startGridW_valid[test_idx]

        endGridU_train = endGridU[train_idx]
        endGridV_train = endGridV[train_idx]
        endGridW_train = endGridW[train_idx]
        endGridU_test = endGridU[test_idx]
        endGridV_test = endGridV[test_idx]
        endGridW_test = endGridW[test_idx]
        endGridU_valid_train = endGridU_valid[train_idx]
        endGridV_valid_train = endGridV_valid[train_idx]
        endGridW_valid_train = endGridW_valid[train_idx]
        endGridU_valid_test = endGridU_valid[test_idx]
        endGridV_valid_test = endGridV_valid[test_idx]
        endGridW_valid_test = endGridW_valid[test_idx]
        
        pfpVars_train = pfpVars[:,0][train_idx]
        pfpVars_test = pfpVars[:,0][test_idx]
    
        trackVars_train = trackVars[train_idx]
        trackVars_test = trackVars[test_idx]

        showerVars_train = showerVars[train_idx]
        showerVars_test = showerVars[test_idx]
    
        y_train = y[train_idx]
        y_test = y[test_idx]

        print('--------------------------------------')    
        print('startGridU_train', startGridU_train.shape)
        print('startGridV_train', startGridV_train.shape)
        print('startGridW_train', startGridW_train.shape)
        print('startGridU_valid_train', startGridU_valid_train.shape)
        print('startGridV_valid_train', startGridV_valid_train.shape)
        print('startGridW_valid_train', startGridW_valid_train.shape)
        print('endGridU_train', endGridU_train.shape)
        print('endGridV_train', endGridV_train.shape)
        print('endGridW_train', endGridW_train.shape)
        print('endGridU_valid_train', endGridU_valid_train.shape)
        print('endGridV_valid_train', endGridV_valid_train.shape)
        print('endGridW_valid_train', endGridW_valid_train.shape)
        print('pfpVars_train', pfpVars_train.shape)
        print('trackVars_train', trackVars_train.shape)
        print('showerVars_train', showerVars_train.shape)
        print('y_train', y_train.shape)

        class_labels = ['Muon', 'Proton', 'Pion', 'Electron', 'Photon']
        particleType_train = np.argmax(y_train, axis=1)
        class_counts_train = np.bincount(particleType_train, minlength=len(class_labels))

        for i in range(len(class_labels)) :
            print(f'n_{class_labels[i]}_train: {class_counts_train[i]}')
        
        print('--------------------------------------')
        print('startGridU_test', startGridU_test.shape)
        print('startGridV_test', startGridV_test.shape)
        print('startGridW_test', startGridW_test.shape)
        print('startGridU_valid_test', startGridU_valid_test.shape)
        print('startGridV_valid_test', startGridV_valid_test.shape)
        print('startGridW_valid_test', startGridW_valid_test.shape)
        print('endGridU_test', endGridU_test.shape)
        print('endGridV_test', endGridV_test.shape)
        print('endGridW_test', endGridW_test.shape)
        print('endGridU_valid_test', endGridU_valid_test.shape)
        print('endGridV_valid_test', endGridV_valid_test.shape)
        print('endGridW_valid_test', endGridW_valid_test.shape)
        print('pfpVars_test', pfpVars_test.shape)
        print('trackVars_test', trackVars_test.shape)
        print('showerVars_test', showerVars_test.shape)
        print('y_test', y_test.shape)

        particleType_test = np.argmax(y_test, axis=1)
        class_counts_test = np.bincount(particleType_test, minlength=len(class_labels))

        for i in range(len(class_labels)) :
            print(f'n_{class_labels[i]}_test: {class_counts_test[i]}')
        
        print('--------------------------------------')    

        prefix = f'{"Contained" if is_contained else "Exiting"}'
        output_file_name = f"{args.output_dir}/{args.file_name.replace('.root', '')}_{prefix}.npz"        

        np.savez(output_file_name,
                 startU_train=startGridU_train, startU_mask_train=startGridU_valid_train,
                 startV_train=startGridV_train, startV_mask_train=startGridV_valid_train,
                 startW_train=startGridW_train, startW_mask_train=startGridW_valid_train,
                 startU_test=startGridU_test, startU_mask_test=startGridU_valid_test,
                 startV_test=startGridV_test, startV_mask_test=startGridV_valid_test,
                 startW_test=startGridW_test, startW_mask_test=startGridW_valid_test,
                 endU_train=endGridU_train, endU_mask_train=endGridU_valid_train,
                 endV_train=endGridV_train, endV_mask_train=endGridV_valid_train,
                 endW_train=endGridW_train, endW_mask_train=endGridW_valid_train,
                 endU_test=endGridU_test, endU_mask_test=endGridU_valid_test,
                 endV_test=endGridV_test, endV_mask_test=endGridV_valid_test,
                 endW_test=endGridW_test, endW_mask_test=endGridW_valid_test,
                 trackVars_train=trackVars_train,
                 trackVars_test=trackVars_test,
                 showerVars_test=showerVars_test,
                 showerVars_train=showerVars_train,
                 y_train=y_train,
                 y_test=y_test)

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

