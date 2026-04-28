import Definitions

class PlotVar :
    def __init__(self, dir_name, tree_name, x_label, y_label, range, n_bins):
        self.dir_name = dir_name
        self.tree_name = tree_name
        self.x_label = x_label
        self.y_label = y_label
        self.range = range
        self.n_bins = n_bins

class PlotDiffVar :
    def __init__(self, dir_name, true_tree_name, reco_tree_name, x_label, y_label, range, n_bins):
        self.dir_name = dir_name
        self.true_tree_name = true_tree_name
        self.reco_tree_name = reco_tree_name
        self.x_label = x_label
        self.y_label = y_label
        self.range = range
        self.n_bins = n_bins  

class SegVar :
    def __init__(self, dir_name, tree_name, options, colors):
        self.dir_name = dir_name
        self.tree_name = tree_name
        self.options = options
        self.colors = colors

class TwoVar :
    def __init__(self, dir_name, plot_var_x, plot_var_y):
        self.dir_name = dir_name
        self.plot_var_x = plot_var_x
        self.plot_var_y = plot_var_y     
        
###############################################
# PlotVar
###############################################        
        
# Event
nu_true_energy = PlotVar('TrueNuEnergy', 'MCNu_Energy', 'True Nu Energy [GeV]', 'Frac. of True Nu', [0,10.0], 40)
nu_vis_true_energy = PlotVar('TrueNuVisEnergy', 'MCNu_VisEnergy', 'True Vis Nu Energy [GeV]', 'Frac. of True Nu', [0,10.0], 40)
nu_vertex_accuracy = PlotVar('NuVertexAccuracy', 'RecoNu_VertexAcc_Pass2', 'NuVertexAccuracy [cm]', 'Frac. of Reco Nu', [0,10.0], 50)
vtx_dr_all = PlotVar('VertexdR_All', 'RecoNu_VertexAcc_Pass2', 'dR [cm]', 'Frac. of True Nu', None, None)    
vtx_dr_only_reco = PlotVar('VertexdR_OnlyReco', 'RecoNu_VertexAcc_Pass2', 'dR [cm]', 'Frac. of Reco Nu', None, None)
Event_MCP_plotting_vars = [nu_true_energy, nu_vis_true_energy]
Event_Reco_plotting_vars = [nu_vertex_accuracy]

# PFP
completeness = PlotVar('Completeness', 'BM_Completeness', 'Completeness', 'Frac. of MCParticles', [0,1.0], 40)
purity = PlotVar('Purity', 'BM_Purity', 'Purity', 'Frac. of MCParticles', [0,1.0], 40)
alt_completeness = PlotVar('AltCompleteness', 'ALT_Completeness', 'AltCompleteness', 'Frac. of MCParticles', [-1.01, 1.0], 40)
alt_purity = PlotVar('AltPurity', 'ALT_Purity', 'AltPurity', 'Frac. of MCParticles', [-1.01, 1.0], 40)
n_mc_hits_2d = PlotVar('NMCHits2D', 'MCP_NMCHits2D', 'NMCHits2D', 'Frac. of PFParticles', [0, 1500], 30)
mc_displacement = PlotVar('Displacement', 'MCP_Displacement', 'MC Displacement [cm]', 'Frac. of PFParticles', [0, 100], 10)
theta_xz = PlotVar('ThetaXZ', 'MCP_TrueThetaXZ', 'ThetaXZ', 'Frac. of PFParticles', [-3.5, 3.5], 25)
theta_yz = PlotVar('ThetaYZ', 'MCP_TrueThetaYZ', 'ThetaYZ', 'Frac. of PFParticles', [-1.6, 1.6], 12)
pfo_energy = PlotVar('TrueEnergy', 'MCP_TrueEnergy', 'True Particle Energy', 'Frac. of PFParticles', [0,3], 50)
true_vis_energy = PlotVar('TrueVisEnergy', 'MCP_TrueVisEnergy', 'True Visible Energy', 'Frac. of PFParticles', [0,2], 20)
pfo_signed_vertex_acc = PlotVar('VertexAcc', 'BM_VertexAcc', 'Signed Vertex deltaR [cm]', 'Frac. of PFParticles', [-25,25], 25)
PFP_MCP_plotting_vars = [completeness, purity, n_mc_hits_2d, true_vis_energy, theta_xz, theta_yz]
PFP_BM_plotting_vars = [pfo_signed_vertex_acc]
PFP_ALT_plotting_vars = [alt_completeness, alt_purity]
PFP_track_shower_plotting_vars = [n_mc_hits_2d, theta_xz, theta_yz]
PFP_efficiency_vars = [n_mc_hits_2d, theta_xz, theta_yz, true_vis_energy, mc_displacement]

# Michel
michel_true_vis_energy = PlotVar('TrueVisEnergy', 'MCP_TrueVisEnergy', 'True Visible Energy', 'Frac. of Michels', [0,0.1], 20)
michel_n_mc_hits_2d = PlotVar('NMCHits2D', 'MCP_NMCHits2D', 'Michel NMCHits2D', 'Frac. of Michels', [Definitions.MIN_TOTAL_HITS, 150], 15)
michel_completeness = PlotVar('Completeness', 'BM_Completeness', 'Completeness', 'Frac. of Michels', [0,1.0], 20)
michel_purity = PlotVar('Purity', 'BM_Purity', 'Purity', 'Frac. of Michels', [0,1.0], 20)
Michel_MCP_plotting_vars = [michel_n_mc_hits_2d, michel_completeness, michel_purity, michel_true_vis_energy]
Michel_track_shower_vars = [michel_n_mc_hits_2d]
Michel_efficiency_vars = [michel_n_mc_hits_2d, michel_true_vis_energy]

# Track
track_endpoint_n_mc_hits = PlotVar('EndpointNMCHits', 'MCP_EndpointsMCHits', 'Endpoint nMCHits', 'Frac. of PFParticles', [0,50], 20)
track_endpoint_completeness = PlotVar('EndpointCompleteness', 'BM_EndpointCompleteness', 'Endpoint Completeness', 'Frac. of PFParticles', [-1.5,1.0], 50)
track_endpoint_purity = PlotVar('EndpointPurity', 'BM_EndpointPurity', 'EndpointPurity', 'Frac. of PFParticles', [-1.5,1.0], 50)
track_signed_endpoint_acc = PlotVar('EndpointAcc', 'BM_EndpointAcc', 'Signed Endpoint deltaR [cm]', 'Frac. of PFParticles', [-25,25], 25)
Track_MCP_plotting_vars = [track_endpoint_completeness, track_endpoint_purity, track_endpoint_n_mc_hits]
Track_BM_plotting_vars = [track_signed_endpoint_acc]

# Shower
shower_initial_MC_hits = PlotVar('InitialMCHits2D', 'MCP_InitialMCHits', 'Initial # MC 2D Hits', 'Frac. of True Showers', [0,150], 30)
shower_initial_PFP_hits = PlotVar('InitialPfoHits2D', 'BM_InitialPfoHits', 'Initial # Pfo 2D Hits', 'Frac. of True Showers', [0,150], 30) # -1 == no MC hits in initial region
shower_initial_completeness = PlotVar('InitialCompleteness', 'BM_InitialCompleteness', 'Initial Completeness', 'Frac. of True Showers', [-1.05,1.0], 20) # -1 == no MC hits in initial region
shower_initial_purity = PlotVar('InitialPurity', 'BM_InitialPurity', 'Initial Purity', 'Frac. of True Showers', [-1.05,1.0], 20)                   
shower_dir_acc = PlotVar('InitialDirAccuracy', 'BM_DirAcc', 'True-Reco Dir Opening Angle [radians]', 'Frac. of Reco Showers', [-1.1, 3.2], 50)  
shower_moliere = PlotVar('MoliereRadius', 'BM_MoliereRadius', 'Moliere Radius', 'Frac. of Reco Showers', [-1.0, 20], 22)  
Shower_MCP_plotting_vars = [shower_initial_MC_hits, shower_initial_PFP_hits, shower_initial_completeness, shower_initial_purity]
Shower_BM_plotting_vars = [shower_dir_acc, shower_moliere]

###############################################
# PlotDiffVar
############################################### 

# # PFP
length_diff_var = PlotDiffVar('DeltaLength', 'MCP_Length', 'BM_Length', 'TrueRecoLength', 'Frac. of PFParticles', [-100, 100], 50)
displacement_diff_var = PlotDiffVar('DeltaDisplacement', 'MCP_Displacement', 'BM_Displacement', 'True-Reco Displacement', 'Frac. of PFParticles', [-20, 20], 20)
PFP_diff_plotting_vars = [length_diff_var, displacement_diff_var]

# Shower
core_length_diff = PlotDiffVar('DeltaCoreLengthFromW', 'MCP_TrueCoreLengthFromW', 'BM_RecoCoreLength', 'True-Reco Core Length (from W) [cm]', 'Frac. of Reco Showers', [-50, 50], 50)
Shower_diff_plotting_vars = [core_length_diff]

# Event
nu_vtx_delta_x = PlotDiffVar('VertexDeltaX', 'MCNu_VertexX', 'RecoNu_VertexX', 'TrueX-RecoX [cm]', 'Frac. of Reco Nu', [-10,10.0], 100)
nu_vtx_delta_y = PlotDiffVar('VertexDeltaY', 'MCNu_VertexY', 'RecoNu_VertexY', 'TrueY-RecoY [cm]', 'Frac. of Reco Nu', [-10,10.0], 100)
nu_vtx_delta_z = PlotDiffVar('VertexDeltaZ', 'MCNu_VertexZ', 'RecoNu_VertexZ', 'TrueZ-RecoZ [cm]', 'Frac. of Reco Nu', [-10,10.0], 100)
Event_diff_plotting_vars = [nu_vtx_delta_x, nu_vtx_delta_y, nu_vtx_delta_z]

###############################################
# SegVar
############################################### 

# ALT
alt_pfp_seg_var = SegVar('PDG', 'ALT_PDG', [13, 2212, 211, 777, 22, 11, 111], ["Blue", "tab:green", "tab:pink", 'tab:purple', "tab:orange", "Red", "tab:olive"])
alt_is_up_hierarchy_seg_var = SegVar('IsUpstreamHierarchy', 'ALT_IsUpstreamHierarchy', [-1, 0, 1], ["tab:gray", "red", "green"]) 
alt_is_same_mc_var = SegVar('IsSameMC', 'ALT_IsSameMC', [-1, 0, 1], ["tab:gray", "red", "green"]) 
ALT_seg_vars = [alt_pfp_seg_var, alt_is_up_hierarchy_seg_var, alt_is_same_mc_var]

###############################################
# TwoVar
###############################################
# PFP
pfp_completeness_true_vis_energy = TwoVar('Completeness_VisEnergy', true_vis_energy, completeness)
pfp_completeness_n_mc_hits = TwoVar('Completeness_NMCHits2D', n_mc_hits_2d, completeness)
pfp_purity_true_vis_energy = TwoVar('Purity_VisEnergy', true_vis_energy, purity)
pfp_purity_n_mc_hits = TwoVar('Purity_NMCHits2D', n_mc_hits_2d, purity)

PFP_profile_vars = [pfp_completeness_true_vis_energy, pfp_completeness_n_mc_hits,
                    pfp_purity_true_vis_energy, pfp_purity_n_mc_hits]

###############################################
# 2D Var
###############################################
# PFP
completeness_purity_var = TwoVar('Completeness_Purity', completeness, purity)
PFP_2D_vars = [completeness_purity_var]
