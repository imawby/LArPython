import numpy as np
import uproot
from tensorflow.keras.utils import to_categorical
import Normalisation
import MapKeys

########################################################################################################

INVALID_VALUE = -0.5

########################################################################################################

def readTree(args) :
    print('Reading trees. This may take a while...')
    
    branch_names = [
        "StartGridU", "StartGridV", "StartGridW",
        "EndGridU", "EndGridV", "EndGridW",
        "PFPN2DHits",
        "NTrackChildren", "NShowerChildren",
        "NGrandChildren", "NChildHits",
        "ChildEnergy", "ChildTrackScore",
        "TrackLength", "TrackWobble",
        "PFPTrackShowerScore",
        "TrackMomComparison",
        "TrackDistFromEdge",
        "ShowerDisplacement",
        "ShowerDCA",
        "ShowerTrackStubLength",
        "ShowerFromParentAvSep",
        "ShowerFromParentChargeAsym",
        "TruePDG",
        "IsPrimary", "IsDeltaRay", "IsMichel"
    ]
    
    file_name = f'{args.input_dir}/{args.file_name}'
    with uproot.open(file_name) as treeFile:
        tree = treeFile["ivyTrain/ivysaur"]
        branches = tree.arrays(expressions=branch_names, library="np")
    
    # Grid lists
    gridMap = {}
    gridMap["startU"] = branches['StartGridU']
    gridMap["startV"] = branches['StartGridV']
    gridMap["startW"] = branches['StartGridW']
    gridMap["endU"] = branches['EndGridU']
    gridMap["endV"] = branches['EndGridV']
    gridMap["endW"] = branches['EndGridW']
    # PFPVar lists
    varMap = {}
    varMap["nHits2D"] = branches['PFPN2DHits']
    varMap["trackScore"] = branches['PFPTrackShowerScore']
    varMap["isPrimary"] = branches['IsPrimary']
    # TrackVar lists
    varMap["nTrackChildren"] = branches['NTrackChildren']
    varMap["nShowerChildren"] = branches['NShowerChildren']
    varMap["nGrandChildren"] = branches['NGrandChildren']
    varMap["nChildHits"] = branches['NChildHits']
    varMap["childEnergy"] = branches['ChildEnergy']
    varMap["childTrackScore"] = branches['ChildTrackScore']
    varMap["trackLength"] = branches['TrackLength']
    varMap["trackWobble"] = branches['TrackWobble']
    varMap["momComparison"] = branches['TrackMomComparison']
    varMap["distToEdge"] = branches['TrackDistFromEdge']
    # ShowerVar lists
    varMap["displacement"] = branches['ShowerDisplacement']
    varMap["dca"] = branches['ShowerDCA']
    varMap["trackStubLength"] = branches['ShowerTrackStubLength']
    varMap["fromParentAvSep"] = branches['ShowerFromParentAvSep']
    varMap["fromParentChargeAsym"] = branches['ShowerFromParentChargeAsym']
    # Truth
    particlePDG = branches['TruePDG']
    isDeltaRay = branches['IsDeltaRay']
    isMichel = branches['IsMichel']
    # Misc
    distToEdge = branches['TrackDistFromEdge']
    isPrimary = branches['IsPrimary']

    del branches
    
    nEntries = particlePDG.shape[0]

    # Only collect taget PDGs
    target_mask = np.isin(np.abs(particlePDG), args.pdgs)
    for key in gridMap.keys():
        gridMap[key] = gridMap[key][target_mask]
    for key in varMap.keys():
        varMap[key] = varMap[key][target_mask] 
    particlePDG = particlePDG[target_mask]
    isDeltaRay = isDeltaRay[target_mask]
    isMichel = isMichel[target_mask]
    distToEdge = distToEdge[target_mask]
    isPrimary = isPrimary[target_mask]
    nEntries = len(particlePDG) 
    
    # Handle grids
    grid_keys = list(gridMap.keys())
    for key in grid_keys:
        # Work out validity (invalid = 0)
        gridMap[key + '_mask'] = gridMap[key] > 1e-7
        # Log energy values
        gridMap[key][gridMap[key + '_mask']] = np.log1p(gridMap[key][gridMap[key + '_mask']])
        print(f'{key} mean: {np.mean(gridMap[key][gridMap[key + "_mask"]]):.4f}')
        print(f'{key} std: {np.std(gridMap[key][gridMap[key + "_mask"]]):.4f}')
        # Normalise
        gridMap[key][gridMap[key + '_mask']] = (gridMap[key][gridMap[key + '_mask']] - Normalisation.grid_mean) / Normalisation.grid_std
    
    # Handle vars
    var_keys = list(varMap.keys())
    for key in var_keys:
        if (key + '_mask') in MapKeys.VAR_KEYS :
            # Work out validity (invalid = -1)
            varMap[key + '_mask'] = varMap[key] > INVALID_VALUE
            # Normalise
            if key in Normalisation.varMean.keys() and key in Normalisation.varStd.keys() :
                print(f'{key} mean: {np.mean(varMap[key][varMap[key + "_mask"]]):.4f}')
                print(f'{key} std: {np.std(varMap[key][varMap[key + "_mask"]]):.4f}')
                varMap[key][varMap[key + '_mask']] = (varMap[key][varMap[key + '_mask']] - Normalisation.varMean[key]) / Normalisation.varStd[key]
        else :
            # Normalise
            if key in Normalisation.varMean.keys() and key in Normalisation.varStd.keys() :
                print(f'{key} mean: {np.mean(varMap[key]):.4f}')
                print(f'{key} std: {np.std(varMap[key]):.4f}')
                varMap[key] = (varMap[key] - Normalisation.varMean[key]) / Normalisation.varStd[key] 

    # Convert to expected format
    dimensions = args.dimensions
    for key in MapKeys.GRID_KEYS:
        gridMap[key] = gridMap[key].astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    for key in MapKeys.VAR_KEYS:
        varMap[key] = varMap[key].astype(np.float32).reshape((nEntries, 1))
    particlePDG = particlePDG.reshape((nEntries, 1))
    isDeltaRay = isDeltaRay.reshape((nEntries, 1))
    isMichel = isMichel.reshape((nEntries, 1))

    trackVars = np.concatenate((varMap["nTrackChildren"],
                                varMap["nTrackChildren_mask"],
                                varMap["nShowerChildren"],
                                varMap["nShowerChildren_mask"],
                                varMap["nGrandChildren"],
                                varMap["nGrandChildren_mask"],
                                varMap["nChildHits"],
                                varMap["nChildHits_mask"],
                                varMap["childEnergy"],
                                varMap["childEnergy_mask"],
                                varMap["childTrackScore"],
                                varMap["childTrackScore_mask"],
                                varMap["trackLength"],
                                varMap["trackLength_mask"],
                                varMap["trackWobble"],
                                varMap["trackWobble_mask"],
                                varMap["momComparison"],
                                varMap["momComparison_mask"],
                                varMap["nHits2D"],
                                varMap["trackScore"],
                                varMap["distToEdge"],
                                varMap["isPrimary"]), axis=1)
    showerVars = np.concatenate((varMap["displacement"],
                                 varMap["displacement_mask"],
                                 varMap["dca"],
                                 varMap["dca_mask"],
                                 varMap["trackStubLength"],
                                 varMap["trackStubLength_mask"],
                                 varMap["fromParentAvSep"],
                                 varMap["fromParentAvSep_mask"],
                                 varMap["fromParentChargeAsym"],
                                 varMap["fromParentChargeAsym_mask"]), axis=1)

    # muons = 0, protons = 1, pions = 2, electrons = 3, michel/DR = 4, photons = 5
    lowEnergy_mask = ((abs(particlePDG) == 11) & ((isDeltaRay == 1) | (isMichel == 1)))
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[(abs(particlePDG) == 11) & (~lowEnergy_mask)] = 3
    particlePDG[(abs(particlePDG) == 11) & (lowEnergy_mask)] = 4
    particlePDG[abs(particlePDG) == 22] = 5
    y = to_categorical(particlePDG, 6)
    
    return gridMap, trackVars, showerVars, y, distToEdge, isPrimary

