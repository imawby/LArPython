# Detector
class DetectorConfig :
    def __init__(self, energy_corr, fv, cvn, izzle, ivysaurus) :
        self.energy_corr = energy_corr
        self.fv = fv
        self.cvn = cvn
        self.izzle = izzle
        self.ivysaurus = ivysaurus

# Energy correction factors
dune_hd_fd_energy_corr = {
    "GradTrkMomRange":0.99,   # gradient for y=mx+c momentum correction for (assumed to be) CCnumu interactions with stopping muons
    "IntTrkMomRange":0.00,    # intercept ''
    "GradTrkMomMCS":0.76,     # gradient for y=mx+c momentum correction for (assumed to be) CCnumu interactions with exiting muons
    "IntTrkMomMCS":0.36,      # intercept ''
    "GradNuMuHadEnCont":0.52, # gradient for y=mx+c hadronic energy correction for (assumed to be) CCnumu interactions with stopping muons
    "IntNuMuHadEnCont":0.00,  # intercept ''
    "GradNuMuHadEnExit":0.56, # gradient for y=mx+c hadronic energy correction for (assumed to be) CCnumu interactions with exiting muons
    "IntNuMuHadEnExit":0.00,  # intercept ''
    "GradShwEnergy":0.91,     # gradient for y=mx+c electron energy correction for (assumed to be) CCnue interactions
    "IntShwEnergy":0.03,      # intercept ''
    "GradNuEHadEn":0.51,      # gradient for y=mx+c hadronic energy correction for (assumed to be) CCnue interactions
    "IntNuEHadEn":0.04        # intercept ''
}
dune_vd_fd_energy_corr = {
    "GradTrkMomRange":0.99,
    "IntTrkMomRange":0.00,
    "GradTrkMomMCS":0.76,
    "IntTrkMomMCS":0.36,
    "GradNuMuHadEnCont":0.52,
    "IntNuMuHadEnCont":0.00,
    "GradNuMuHadEnExit":0.56,
    "IntNuMuHadEnExit":0.00,
    "GradShwEnergy":0.91,
    "IntShwEnergy":0.03,
    "GradNuEHadEn":0.51,
    "IntNuEHadEn":0.04
}

# fiducial volume cuts
dune_hd_fd_fv = {
    "MinX":(-360.0 + 50.0),
    "MaxX":(360.0 - 50.0),
    "MinY":(-600.0 + 50.0),
    "MaxY":(600.0 - 50.0),
    "MinZ":(0 + 50.0),
    "MaxZ":(1394.0 - 150.0)
}

# CVN selection
# DUNE CVN is a CNN which predicts neutrino interaction type (CCnumu, CCnue, CCnutau, NC)
# CCnumu/nue selection is simply a cut on the corresponding skills
dune_hd_fd_cvn = {
    "nue_cvn_score_cut":0.85,
    "numu_cvn_score_cut":0.50
}

# Izzle selection
# CCnumu/nue interactions are selected based on the PID scores of the candidate leading leptons
# Electron PID performed by enhanced & backup pandRizzle (the former/latter is used if connection pathway metadata has/hasn't been filled)
# Muon PID performed by pandizzle
dune_hd_fd_izzle = {
    "nue_enhanced_pandrizzle_score_cut":0.4,
    "nue_enhanced_pandrizzle_hit_cut":100,
    "nue_backup_pandrizzle_score_cut":1.0,
    "nue_backup_pandrizzle_hit_cut":25,
    "nue_pandizzle_score_cut":0.7,
    "numu_pandizzle_score_cut":0.4
}

# Ivysaurus selection
# CCnumu/nue interactions are selected based on the PID scores of the candidate leading leptons
# PID performed by Ivysaurus, a CNN based network which operates on the start and end regions of a particle
dune_hd_fd_ivysaurus = {
    "nue_ivy_electron_score_cut":0.87,
    "nue_ivy_electron_hit_cut":100,
    "nue_ivy_muon_score_cut":0.50,
    "numu_ivy_muon_score_cut":0.40    
}
 
