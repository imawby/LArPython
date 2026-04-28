import Signal

#####################################################################################################################
#####################################################################################################################

# CVN
CVN_NUE_CUT = 0.85
CVN_NUMU_CUT = 0.50

# Pandrizzle/Pandizzle
NUE_ENHANCED_PANDRIZZLE_CUT = 0.4     # thesis == 0.025
NUE_ENHANCED_PANDRIZZLE_HIT_CUT = 100 # thesis == 100
NUE_BACKUP_PANDRIZZLE_CUT = 1.0       # thesis == 0.625
NUE_BACKUP_PANDRIZZLE_HIT_CUT = 25    # thesis == 0.625
NUE_PANDIZZLE_CUT = 0.7               # thesis == 0.85
NUMU_PANDIZZLE_CUT = 0.4              # thesis == 0.025

# Ivysaurus
NUMU_IVY_CUT = 0.4
NUE_ELECTRON_IVY_CUT = 0.87
NUE_ELECTRON_HIT_CUT = 100
NUE_MUON_IVY_CUT = 0.5

#####################################################################################################################
#####################################################################################################################

def PassCCNueSelection_CVN(nusel_branches) :
    return (Signal.IsInFiducialVolume(nusel_branches, False)) & (nusel_branches['CVNResultNue'] > CVN_NUE_CUT)

#####################################################################################################################
#####################################################################################################################

def PassCCNumuSelection_CVN(nusel_branches) :
    return (Signal.IsInFiducialVolume(nusel_branches, False)) & (nusel_branches['CVNResultNumu'] > CVN_NUMU_CUT)

#####################################################################################################################
#####################################################################################################################

def PassCCNuePandrizzleSelection(nusel_branches, nue_enhanced_pandrizzle_cut=NUE_ENHANCED_PANDRIZZLE_CUT,
                                 nue_backup_pandrizzle_cut=NUE_BACKUP_PANDRIZZLE_CUT, nue_pandizzle_cut=NUE_PANDIZZLE_CUT) :

    pass_enhanced = (nusel_branches['SelShowerEnhancedPandrizzleScore'] > nue_enhanced_pandrizzle_cut) & (nusel_branches['SelShowerPandrizzleNHits'] > NUE_ENHANCED_PANDRIZZLE_HIT_CUT)
    pass_backup = (nusel_branches['SelShowerBackupPandrizzleScore'] > nue_backup_pandrizzle_cut) & (nusel_branches['SelShowerPandrizzleNHits'] > NUE_BACKUP_PANDRIZZLE_HIT_CUT)           
    sel_shower = pass_enhanced | pass_backup                    
    
    return Signal.IsInFiducialVolume(nusel_branches, False) & sel_shower & (nusel_branches['SelTrackPandizzleScore'] < nue_pandizzle_cut)

#####################################################################################################################
#####################################################################################################################

def PassCCNumuPandizzleSelection(nusel_branches, numu_pandizzle_cut=NUMU_PANDIZZLE_CUT) :

    does_pass_ccnue = PassCCNuePandrizzleSelection(nusel_branches)
    
    return (~does_pass_ccnue) & Signal.IsInFiducialVolume(nusel_branches, False) & (nusel_branches['SelTrackPandizzleScore'] > numu_pandizzle_cut)    

#####################################################################################################################
#####################################################################################################################

def PassCCNumuIvysaurusSelection(nusel_branches, numu_ivy_cut=NUMU_IVY_CUT) :    
    return Signal.IsInFiducialVolume(nusel_branches, False) & (nusel_branches['SelTrackIvysaurusScore'] > numu_ivy_cut)    

#####################################################################################################################
#####################################################################################################################

def PassCCNueIvysaurusSelection(nusel_branches, nue_electron_ivy_cut=NUE_ELECTRON_IVY_CUT, nue_muon_ivy_cut=NUE_MUON_IVY_CUT) :    
    return Signal.IsInFiducialVolume(nusel_branches, False) & (nusel_branches['SelShowerIvysaurusScore'] > nue_electron_ivy_cut) & (nusel_branches['SelTrackIvysaurusScore'] < nue_muon_ivy_cut) & (nusel_branches['SelShowerIvysaurusNHits'] > NUE_ELECTRON_HIT_CUT)
