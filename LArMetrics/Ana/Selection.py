import Signal

#####################################################################################################################

def PassCCNueSelection_CVN(nusel_branches, det_config) :
    return (Signal.IsInFiducialVolume(nusel_branches, det_config.fv, False)) & (nusel_branches['CVNResultNue'] > det_config.cvn["nue_cvn_score_cut"])

#####################################################################################################################

def PassCCNumuSelection_CVN(nusel_branches, det_config) :
    return (Signal.IsInFiducialVolume(nusel_branches, det_config.fv, False)) & (nusel_branches['CVNResultNumu'] > det_config.cvn["numu_cvn_score_cut"])

#####################################################################################################################

def PassCCNueSelection_Izzle(nusel_branches, det_config, nue_enhanced_pandrizzle_score_cut="",
                                 nue_backup_pandrizzle_score_cut="", nue_pandizzle_score_cut="") :
    if (nue_enhanced_pandrizzle_score_cut == "") : nue_enhanced_pandrizzle_score_cut = det_config.izzle["nue_enhanced_pandrizzle_score_cut"]
    if (nue_backup_pandrizzle_score_cut == "") : nue_backup_pandrizzle_score_cut = det_config.izzle["nue_backup_pandrizzle_score_cut"]
    if (nue_pandizzle_score_cut == "") : nue_pandizzle_score_cut = det_config.izzle["nue_pandizzle_score_cut"]
    
    pass_enhanced = (nusel_branches['SelShowerEnhancedPandrizzleScore'] > nue_enhanced_pandrizzle_score_cut) &\
                    (nusel_branches['SelShowerPandrizzleNHits'] > det_config.izzle["nue_enhanced_pandrizzle_hit_cut"])   
    pass_backup = (nusel_branches['SelShowerBackupPandrizzleScore'] > nue_backup_pandrizzle_score_cut) &\
                  (nusel_branches['SelShowerPandrizzleNHits'] > det_config.izzle["nue_backup_pandrizzle_hit_cut"])           
    sel_shower = pass_enhanced | pass_backup                    
    
    return Signal.IsInFiducialVolume(nusel_branches, det_config.fv, False) & sel_shower &\
        (nusel_branches['SelTrackPandizzleScore'] < nue_pandizzle_score_cut)

#####################################################################################################################

def PassCCNumuSelection_Izzle(nusel_branches, det_config, numu_pandizzle_score_cut="") :
    if (numu_pandizzle_score_cut == "") : numu_pandizzle_score_cut = det_config.izzle["numu_pandizzle_score_cut"]
    does_pass_ccnue = PassCCNueSelection_Izzle(nusel_branches, det_config)
    
    return (~does_pass_ccnue) & Signal.IsInFiducialVolume(nusel_branches, det_config.fv, False) &\
        (nusel_branches['SelTrackPandizzleScore'] > numu_pandizzle_score_cut)    

#####################################################################################################################

def PassCCNumuSelection_Ivysaurus(nusel_branches, det_config, numu_ivy_muon_score_cut="") :
    if (numu_ivy_muon_score_cut == "") : numu_ivy_muon_score_cut = det_config.ivysaurus["numu_ivy_muon_score_cut"]
    return Signal.IsInFiducialVolume(nusel_branches, det_config.fv, False) & (nusel_branches['SelTrackIvysaurusScore'] > numu_ivy_muon_score_cut)    

#####################################################################################################################

def PassCCNueSelection_Ivysaurus(nusel_branches, det_config, nue_ivy_electron_score_cut="", nue_ivy_muon_score_cut="") :
    if (nue_ivy_electron_score_cut == "") : nue_ivy_electron_score_cut = det_config.ivysaurus["nue_ivy_electron_score_cut"]
    if (nue_ivy_muon_score_cut == "") : nue_ivy_muon_score_cut = det_config.ivysaurus["nue_ivy_muon_score_cut"]
    
    return Signal.IsInFiducialVolume(nusel_branches, det_config.fv, False) & (nusel_branches['SelShowerIvysaurusScore'] > nue_ivy_electron_score_cut) &\
        (nusel_branches['SelTrackIvysaurusScore'] < nue_ivy_muon_score_cut) & (nusel_branches['SelShowerIvysaurusNHits'] > det_config.ivysaurus["nue_ivy_electron_hit_cut"])

