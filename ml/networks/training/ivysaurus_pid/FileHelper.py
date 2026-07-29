import numpy as np
import uproot

from tensorflow.keras.utils import to_categorical

import Normalisation

########################################################################################################

def readTree(file_name, dimensions, detector) :

    print('Reading trees. This may take a while...')
    
    branch_names = [
        "StartGridU", "StartGridV", "StartGridW",
        "EndGridU", "EndGridV", "EndGridW",
        "PFPN2DHits",
        "RecoEndX", "RecoEndY", "RecoEndZ",        
        "TrueEndX", "TrueEndY", "TrueEndZ",
        "NTrackChildren", "NShowerChildren",
        "NGrandChildren", "NChildHits",
        "ChildEnergy", "ChildTrackScore",
        "TrackLength", "TrackWobble",
        "PFPTrackShowerScore",
        "TrackMomComparison",
        "ShowerDisplacement",
        "ShowerDCA",
        "ShowerTrackStubLength",
        "ShowerNuVertexAvSeparation",
        "ShowerNuVertexChargeAsymmetry",
        "TruePDG",
        "IsPrimary"
    ]

    with uproot.open(file_name) as treeFile:
        #tree = treeFile["ivysaur"]
        tree = treeFile["ivysaur"]
        branches = tree.arrays(expressions=branch_names, library="np")
    
    # Grid lists
    startGridU = branches['StartGridU']
    startGridV = branches['StartGridV']
    startGridW = branches['StartGridW']
    endGridU = branches['EndGridU']
    endGridV = branches['EndGridV']
    endGridW = branches['EndGridW']
    # PFPVar lists
    nHits2D = branches['PFPN2DHits']
    trackScore = branches['PFPTrackShowerScore']            
    # TrackVar lists
    nTrackChildren = branches['NTrackChildren']
    nShowerChildren = branches['NShowerChildren']
    nGrandChildren = branches['NGrandChildren']
    nChildHits = branches['NChildHits']
    childEnergy = branches['ChildEnergy']
    childTrackScore = branches['ChildTrackScore']
    trackLength = branches['TrackLength']
    trackWobble = branches['TrackWobble']
    momComparison = branches['TrackMomComparison']
    # ShowerVar lists  
    displacement = branches['ShowerDisplacement']
    dca = branches['ShowerDCA']
    trackStubLength = branches['ShowerTrackStubLength']
    nuVertexAvSeparation = branches['ShowerNuVertexAvSeparation']
    nuVertexChargeAsymmetry = branches['ShowerNuVertexChargeAsymmetry']
    # Truth
    particlePDG = branches['TruePDG']
    # Misc
    endX = branches['RecoEndX']
    endY = branches['RecoEndY']
    endZ = branches['RecoEndZ']
    isPrimary = branches['IsPrimary']
    del branches

    dist_to_edge = np.min(
        np.stack([
            np.abs(endX - detector['MinX']),
            np.abs(endX - detector['MaxX']),
            np.abs(endY - detector['MinY']),
            np.abs(endY - detector['MaxY']),
            np.abs(endZ - detector['MinZ']),
            np.abs(endZ - detector['MaxZ'])
        ], axis=0),
        axis=0)


    print('dist_to_edge:', dist_to_edge.shape)
    
    nEntries = particlePDG.shape[0]

    ###################################
    # Only keep primaries
    ###################################
    primaryMask = (abs(particlePDG) != 321) #& (isPrimary != 1) & (abs(particlePDG) != 2212)
    #((abs(particlePDG) == 11) | (abs(particlePDG) == 13)) #| (abs(particlePDG) == 211) | (abs(particlePDG) == 22))
    #& (isPrimary == 1)  & ((abs(particlePDG) == 11) | (abs(particlePDG) == 13) | (abs(particlePDG) == 211) | (abs(particlePDG) == 22))

    startGridU = startGridU[primaryMask]
    startGridV = startGridV[primaryMask]
    startGridW = startGridW[primaryMask]
    endGridU = endGridU[primaryMask]
    endGridV = endGridV[primaryMask]
    endGridW = endGridW[primaryMask]
    nHits2D = nHits2D[primaryMask]
    trackScore = trackScore[primaryMask]
    dist_to_edge = dist_to_edge[primaryMask]    
    endX = endX[primaryMask]
    endY = endY[primaryMask]
    endZ = endZ[primaryMask]
    nTrackChildren = nTrackChildren[primaryMask]
    nShowerChildren = nShowerChildren[primaryMask]
    nGrandChildren = nGrandChildren[primaryMask]
    nChildHits = nChildHits[primaryMask]
    childEnergy = childEnergy[primaryMask]
    childTrackScore = childTrackScore[primaryMask]
    trackLength = trackLength[primaryMask]
    trackWobble = trackWobble[primaryMask]
    momComparison = momComparison[primaryMask]    
    displacement = displacement[primaryMask]
    dca = dca[primaryMask]
    trackStubLength = trackStubLength[primaryMask]
    nuVertexAvSeparation = nuVertexAvSeparation[primaryMask]
    nuVertexChargeAsymmetry = nuVertexChargeAsymmetry[primaryMask]
    particlePDG = particlePDG[primaryMask]
    isPrimary = isPrimary[primaryMask]
    nEntries = len(particlePDG)

    print("After primary selection:", nEntries)
    # Refinement of the particlePDG vector
    print('We have ', str(nEntries), ' PFParticles overall!')
    print('nMuons: ', np.count_nonzero(abs(particlePDG) == 13))
    print('nProtons: ', np.count_nonzero(abs(particlePDG) == 2212))    
    print('nPions: ', np.count_nonzero(abs(particlePDG) == 211))     
    print('nKaons: ', np.count_nonzero(abs(particlePDG) == 321))
    print('nElectrons: ', np.count_nonzero(abs(particlePDG) == 11))         
    print('nPhotons: ', np.count_nonzero(abs(particlePDG) == 22))    
    
    # Handle grids
    # Work out validity (invalid = 0)
    startGridU_valid = startGridU > 1e-7
    startGridV_valid = startGridV > 1e-7
    startGridW_valid = startGridW > 1e-7
    endGridU_valid = endGridU > 1e-7
    endGridV_valid = endGridV > 1e-7
    endGridW_valid = endGridW > 1e-7

    # Log energy values
    startGridU[startGridU_valid] = np.log1p(startGridU[startGridU_valid])
    startGridV[startGridV_valid] = np.log1p(startGridV[startGridV_valid])
    startGridW[startGridW_valid] = np.log1p(startGridW[startGridW_valid])
    endGridU[endGridU_valid] = np.log1p(endGridU[endGridU_valid])
    endGridV[endGridV_valid] = np.log1p(endGridV[endGridV_valid])
    endGridW[endGridW_valid] = np.log1p(endGridW[endGridW_valid])

    # Normalise them
    print('--------------------------------------------------')
    print(f'startGridU mean: {np.mean(startGridU[startGridU_valid]):.4f}')
    print(f'startGridU std: {np.std(startGridU[startGridU_valid]):.4f}')
    print('--------------------------------------------------')
    print(f'startGridV mean: {np.mean(startGridV[startGridV_valid]):.4f}')
    print(f'startGridV std: {np.std(startGridV[startGridV_valid]):.4f}')
    print('--------------------------------------------------')
    print(f'startGridW mean: {np.mean(startGridW[startGridW_valid]):.4f}')
    print(f'startGridW std: {np.std(startGridW[startGridW_valid]):.4f}')
    print('--------------------------------------------------')
    print(f'endGridU mean: {np.mean(endGridU[endGridU_valid]):.4f}')
    print(f'endGridU std: {np.std(endGridU[endGridU_valid]):.4f}')
    print('--------------------------------------------------')
    print(f'endGridV mean: {np.mean(endGridV[endGridV_valid]):.4f}')
    print(f'endGridV std: {np.std(endGridV[endGridV_valid]):.4f}')
    print('--------------------------------------------------')
    print(f'endGridW mean: {np.mean(endGridW[endGridW_valid]):.4f}')
    print(f'endGridW std: {np.std(endGridW[endGridW_valid]):.4f}')
    print('--------------------------------------------------')
    startGridU[startGridU_valid] = (startGridU[startGridU_valid] - Normalisation.grid_mean) / Normalisation.grid_std
    startGridV[startGridV_valid] = (startGridV[startGridV_valid] - Normalisation.grid_mean) / Normalisation.grid_std
    startGridW[startGridW_valid] = (startGridW[startGridW_valid] - Normalisation.grid_mean) / Normalisation.grid_std
    endGridU[endGridU_valid] = (endGridU[endGridU_valid] - Normalisation.grid_mean) / Normalisation.grid_std
    endGridV[endGridV_valid] = (endGridV[endGridV_valid] - Normalisation.grid_mean) / Normalisation.grid_std
    endGridW[endGridW_valid] = (endGridW[endGridW_valid] - Normalisation.grid_mean) / Normalisation.grid_std
    
    # PFP vars
    # Normalise them    
    print('--------------------------------------------------')
    print(f'nHits2D mean: {np.mean(nHits2D):.4f}')
    print(f'nHits2D std: {np.std(nHits2D):.4f}')
    print('--------------------------------------------------')
    print(f'trackScore mean: {np.mean(trackScore):.4f}')
    print(f'trackScore std: {np.std(trackScore):.4f}')
    print('--------------------------------------------------')
    print(f'distToEdge mean: {np.mean(dist_to_edge):.4f}')
    print(f'distToEdge std: {np.std(dist_to_edge):.4f}')    
    print('--------------------------------------------------')
    
    nHits2D = (nHits2D - Normalisation.nHits2D_mean) / Normalisation.nHits2D_std
    trackScore = (trackScore - Normalisation.trackScore_mean) /  Normalisation.trackScore_std
    
    # Track vars 
    # Work out validity (invalid = -1)
    nTrackChildren_valid = nTrackChildren > -0.5
    nShowerChildren_valid = nShowerChildren > -0.5
    nGrandChildren_valid = nGrandChildren > -0.5
    nChildHits_valid = nChildHits > -0.5
    childEnergy_valid = childEnergy > -0.5
    childTrackScore_valid = childTrackScore > -0.5
    trackLength_valid = trackLength > -0.5
    trackWobble_valid = trackWobble > -0.5
    momComparison_valid = momComparison > -0.5

    # Normalise
    print('--------------------------------------------------')
    print(f'nTrackChildren mean: {np.mean(nTrackChildren[nTrackChildren_valid]):.4f}')
    print(f'nTrackChildren std: {np.std(nTrackChildren[nTrackChildren_valid]):.4f}')
    print('--------------------------------------------------')
    print(f'nShowerChildren mean: {round(float(np.mean(nShowerChildren[nShowerChildren_valid])), 4)}')
    print(f'nShowerChildren std: {round(float(np.std(nShowerChildren[nShowerChildren_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'nGrandChildren mean: {round(float(np.mean(nGrandChildren[nGrandChildren_valid])), 4)}')
    print(f'nGrandChildren std: {round(float(np.std(nGrandChildren[nGrandChildren_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'nChildHits mean: {round(float(np.mean(nChildHits[nChildHits_valid])), 4)}')
    print(f'nChildHits std: {round(float(np.std(nChildHits[nChildHits_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'childEnergy mean: {round(float(np.mean(childEnergy[childEnergy_valid])), 4)}')
    print(f'childEnergy std: {round(float(np.std(childEnergy[childEnergy_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'childTrackScore mean: {round(float(np.mean(childTrackScore[childTrackScore_valid])), 4)}')
    print(f'childTrackScore std: {round(float(np.std(childTrackScore[childTrackScore_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'trackLength mean: {round(float(np.mean(trackLength[trackLength_valid])), 4)}')
    print(f'trackLength std: {round(float(np.std(trackLength[trackLength_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'trackWobble mean: {round(float(np.mean(trackWobble[trackWobble_valid])), 4)}')
    print(f'trackWobble std: {round(float(np.std(trackWobble[trackWobble_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'momComparison mean: {round(float(np.mean(momComparison[momComparison_valid])), 4)}')
    print(f'momComparison std: {round(float(np.std(momComparison[momComparison_valid])), 4)}')
    print('--------------------------------------------------')
 
    nTrackChildren[nTrackChildren_valid] = (nTrackChildren[nTrackChildren_valid] - Normalisation.nTrackChildren_mean) /  Normalisation.nTrackChildren_std
    nShowerChildren[nShowerChildren_valid] = (nShowerChildren[nShowerChildren_valid] -  Normalisation.nShowerChildren_mean) /  Normalisation.nShowerChildren_std
    nGrandChildren[nGrandChildren_valid] = (nGrandChildren[nGrandChildren_valid] -  Normalisation.nGrandChildren_mean) /  Normalisation.nGrandChildren_std
    nChildHits[nChildHits_valid] = (nChildHits[nChildHits_valid] -  Normalisation.nChildHits_mean) /  Normalisation.nChildHits_std
    childEnergy[childEnergy_valid] = (childEnergy[childEnergy_valid] -  Normalisation.childEnergy_mean) /  Normalisation.childEnergy_std
    childTrackScore[childTrackScore_valid] = (childTrackScore[childTrackScore_valid] -  Normalisation.childTrackScore_mean) /  Normalisation.childTrackScore_std
    trackLength[trackLength_valid] = (trackLength[trackLength_valid] -  Normalisation.trackLength_mean) /  Normalisation.trackLength_std
    trackWobble[trackWobble_valid] = (trackWobble[trackWobble_valid] -  Normalisation.trackWobble_mean) /  Normalisation.trackWobble_std
    momComparison[momComparison_valid] = (momComparison[momComparison_valid] - Normalisation. momComparison_mean) /  Normalisation.momComparison_std

    # Shower vars 
    # Work out validity (invalid = -1)    
    displacement_valid = displacement > -0.5
    dca_valid = dca > -0.5
    trackStubLength_valid = trackStubLength > -0.5
    nuVertexAvSeparation_valid = nuVertexAvSeparation > -0.5
    nuVertexChargeAsymmetry_valid = nuVertexChargeAsymmetry > -0.5

    # Normalise
    print(f'displacement mean: {round(float(np.mean(displacement[displacement_valid])), 4)}')
    print(f'displacement std: {round(float(np.std(displacement[displacement_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'dca mean: {round(float(np.mean(dca[dca_valid])), 4)}')
    print(f'dca std: {round(float(np.std(dca[dca_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'trackStubLength mean: {round(float(np.mean(trackStubLength[trackStubLength_valid])), 4)}')
    print(f'trackStubLength std: {round(float(np.std(trackStubLength[trackStubLength_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'nuVertexAvSeparation mean: {round(float(np.mean(nuVertexAvSeparation[nuVertexAvSeparation_valid])), 4)}')
    print(f'nuVertexAvSeparation std: {round(float(np.std(nuVertexAvSeparation[nuVertexAvSeparation_valid])), 4)}')
    print('--------------------------------------------------')
    print(f'nuVertexChargeAsymmetry mean: {round(float(np.mean(nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid])), 4)}')
    print(f'nuVertexChargeAsymmetry std: {round(float(np.std(nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid])), 4)}')
    print('--------------------------------------------------')

    displacement[displacement_valid] = (displacement[displacement_valid] - Normalisation.displacement_mean) / Normalisation.displacement_std
    dca[dca_valid] = (dca[dca_valid] - Normalisation.dca_mean) / Normalisation.dca_std
    trackStubLength[trackStubLength_valid] = (trackStubLength[trackStubLength_valid] - Normalisation.trackStubLength_mean) / Normalisation.trackStubLength_std
    nuVertexAvSeparation[nuVertexAvSeparation_valid] = (nuVertexAvSeparation[nuVertexAvSeparation_valid] - Normalisation.nuVertexAvSeparation_mean) / Normalisation.nuVertexAvSeparation_std
    nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid] = (nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid] - Normalisation.nuVertexChargeAsymmetry_mean) / Normalisation.nuVertexChargeAsymmetry_std

    # Convert to expected format
    startGridU = startGridU.reshape((nEntries, dimensions, dimensions, 1))
    startGridV = startGridV.reshape((nEntries, dimensions, dimensions, 1))
    startGridW = startGridW.reshape((nEntries, dimensions, dimensions, 1))
    endGridU = endGridU.reshape((nEntries, dimensions, dimensions, 1))
    endGridV = endGridV.reshape((nEntries, dimensions, dimensions, 1))
    endGridW = endGridW.reshape((nEntries, dimensions, dimensions, 1))
    startGridU_valid = startGridU_valid.astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    startGridV_valid = startGridV_valid.astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    startGridW_valid = startGridW_valid.astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    endGridU_valid = endGridU_valid.astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    endGridV_valid = endGridV_valid.astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    endGridW_valid = endGridW_valid.astype(np.float32).reshape((nEntries, dimensions, dimensions, 1))
    nHits2D = nHits2D.reshape((nEntries, 1))
    trackScore = trackScore.reshape((nEntries, 1))
    nTrackChildren = nTrackChildren.reshape((nEntries, 1))
    nShowerChildren = nShowerChildren.reshape((nEntries, 1))
    nGrandChildren = nGrandChildren.reshape((nEntries, 1))
    nChildHits = nChildHits.reshape((nEntries, 1))
    childEnergy = childEnergy.reshape((nEntries, 1))
    childTrackScore = childTrackScore.reshape((nEntries, 1))
    trackLength = trackLength.reshape((nEntries, 1))
    trackWobble = trackWobble.reshape((nEntries, 1))
    momComparison = momComparison.reshape((nEntries, 1))
    nTrackChildren_valid = nTrackChildren_valid.astype(np.float32).reshape((nEntries, 1))
    nShowerChildren_valid = nShowerChildren_valid.astype(np.float32).reshape((nEntries, 1))
    nGrandChildren_valid = nGrandChildren_valid.astype(np.float32).reshape((nEntries, 1))
    nChildHits_valid = nChildHits_valid.astype(np.float32).reshape((nEntries, 1))
    childEnergy_valid = childEnergy_valid.astype(np.float32).reshape((nEntries, 1))
    childTrackScore_valid = childTrackScore_valid.astype(np.float32).reshape((nEntries, 1))
    trackLength_valid = trackLength_valid.astype(np.float32).reshape((nEntries, 1))
    trackWobble_valid = trackWobble_valid.astype(np.float32).reshape((nEntries, 1))
    momComparison_valid = momComparison_valid.astype(np.float32).reshape((nEntries, 1))
    displacement = displacement.reshape((nEntries, 1))
    dca = dca.reshape((nEntries, 1))
    trackStubLength = trackStubLength.reshape((nEntries, 1))
    nuVertexAvSeparation = nuVertexAvSeparation.reshape((nEntries, 1))
    nuVertexChargeAsymmetry = nuVertexChargeAsymmetry.reshape((nEntries, 1))    
    displacement_valid = displacement_valid.astype(np.float32).reshape((nEntries, 1))
    dca_valid = dca_valid.astype(np.float32).reshape((nEntries, 1))
    trackStubLength_valid = trackStubLength_valid.astype(np.float32).reshape((nEntries, 1))
    nuVertexAvSeparation_valid = nuVertexAvSeparation_valid.astype(np.float32).reshape((nEntries, 1))
    nuVertexChargeAsymmetry_valid = nuVertexChargeAsymmetry_valid.astype(np.float32).reshape((nEntries, 1))
    particlePDG = particlePDG.reshape((nEntries, 1))
    endX = endX.reshape((nEntries, 1))
    endY = endY.reshape((nEntries, 1))
    endZ = endZ.reshape((nEntries, 1))
    
    pfpVars = np.concatenate((endX, endY, endZ), axis=1)
    trackVars = np.concatenate((nTrackChildren, nTrackChildren_valid,
                                nShowerChildren, nShowerChildren_valid,
                                nGrandChildren, nGrandChildren_valid,
                                nChildHits, nChildHits_valid,
                                childEnergy, childEnergy_valid,
                                childTrackScore, childTrackScore_valid,
                                trackLength, trackLength_valid,
                                trackWobble, trackWobble_valid,
                                momComparison, momComparison_valid, nHits2D, trackScore), axis=1)
    showerVars = np.concatenate((displacement, displacement_valid,
                                 dca, dca_valid,
                                 trackStubLength, trackStubLength_valid,
                                 nuVertexAvSeparation, nuVertexAvSeparation_valid,
                                 nuVertexChargeAsymmetry, nuVertexChargeAsymmetry_valid), axis=1)
   
    # muons = 0, protons = 1, pions = 2, kaons = 3, electrons = 4, photons = 5
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    # particlePDG[abs(particlePDG) == 321] = 3
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    
    # remove kaons!
    # keepMask = (abs(particlePDG[:,0]) != 321) & (isPrimary == 1)
    # particlePDG = particlePDG[keepMask]
    # startGridU = startGridU[keepMask]
    # startGridV = startGridV[keepMask]
    # startGridW = startGridW[keepMask]
    # startGridU_valid = startGridU_valid[keepMask]
    # startGridV_valid = startGridV_valid[keepMask]
    # startGridW_valid = startGridW_valid[keepMask]
    # endGridU = endGridU[keepMask]
    # endGridV = endGridV[keepMask]
    # endGridW = endGridW[keepMask]
    # endGridU_valid = endGridU_valid[keepMask]
    # endGridV_valid = endGridV_valid[keepMask]
    # endGridW_valid = endGridW_valid[keepMask]        
    # pfpVars = pfpVars[keepMask]
    # trackVars = trackVars[keepMask]
    # showerVars = showerVars[keepMask]
    y = to_categorical(particlePDG, 5)
    
    return startGridU, startGridU_valid, startGridV, startGridV_valid, startGridW, startGridW_valid, endGridU, endGridU_valid, endGridV, endGridV_valid, endGridW, endGridW_valid, pfpVars, trackVars, showerVars, y

