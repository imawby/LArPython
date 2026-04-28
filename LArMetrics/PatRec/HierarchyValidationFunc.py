import awkward as ak
import Definitions

##############################################################################################
##############################################################################################

def run_hierarchy_validation(plot_dir, target_mask, reco_mask, int_masks, tier_masks, pdg_masks, hierarchy_branches, pfp_branches) :
    create_table_metrics(plot_dir, target_mask, reco_mask, int_masks, tier_masks, pdg_masks, hierarchy_branches, pfp_branches, demand_parent_has_match=True, split_by_pdg=True)
    create_table_metrics(plot_dir, target_mask, reco_mask, int_masks, tier_masks, pdg_masks, hierarchy_branches, pfp_branches, demand_parent_has_match=True, split_by_pdg=False)
    create_table_metrics(plot_dir, target_mask, reco_mask, int_masks, tier_masks, pdg_masks, hierarchy_branches, pfp_branches, demand_parent_has_match=False, split_by_pdg=True)
    create_table_metrics(plot_dir, target_mask, reco_mask, int_masks, tier_masks, pdg_masks, hierarchy_branches, pfp_branches, demand_parent_has_match=False, split_by_pdg=False)

##############################################################################################
##############################################################################################

def create_table_metrics(plot_dir, target_mask, reco_mask, int_masks, tier_masks,  pdg_masks, hierarchy_branches, pfp_branches, demand_parent_has_match, split_by_pdg) :
    # Cache awkward arrays locally
    mc_tier      = hierarchy_branches['MC_HierarchyTier']
    bm_tier      = hierarchy_branches['BM_HierarchyTier']
    mc_parent    = hierarchy_branches['MC_ParentIndex']
    bm_parent    = hierarchy_branches['BM_ParentIndex']
    mc_has_match = target_mask & reco_mask
    
    for int_type in Definitions.ints :
        file_name = f'HierarchyTable_{Definitions.int_file_strings[int_type]}' + ('_PDG' if split_by_pdg else '') + ('_ParentRecod' if demand_parent_has_match else '')
        
        with open(f'{plot_dir}{file_name}.txt', "w") as f:
            print("DEMAND_PARENT_HAS_MATCH =", demand_parent_has_match, file=f)
            print("SPLIT_BY_PDG =", split_by_pdg, file=f)
            print("", file=f)
        
            # PDG
            pdgs = Definitions.pdgs if split_by_pdg else [-1]
        
            for pdg in pdgs :
        
                pdg_mask = pdg_masks[pdg] if split_by_pdg else ak.ones_like(int_masks[int_type])
                pdg_string = Definitions.pdg_strings[pdg] if split_by_pdg else 'All PDG'
            
                print(f'{Definitions.int_strings[int_type]} - {pdg_string}', file=f)
                print('------------------------------------------------------------------------------------', file=f)
                print('           | Correct Parent | False Primary | Wrong Parent | Parent Not Best Match |', file=f)
                print('------------------------------------------------------------------------------------', file=f)    
            
                for tier in Definitions.tiers :
                    target_mask = mc_has_match & tier_masks[tier] & int_masks[int_type] & pdg_mask
                    primary_reco_mask = target_mask & (bm_tier == 1)
                    other_reco_mask = target_mask & (bm_tier != 1)
                
                    # If we're looking at the correctness of parent-child links,
                    # do we want to demand that the parent is reconstructed?
                    if ((tier != 0) & demand_parent_has_match) :
                        other_reco_mask = other_reco_mask & mc_has_match[mc_parent]
                        
                    n_other = ak.sum(other_reco_mask) + ak.sum(primary_reco_mask)
                
                    # Correct non-nu parent-child links?
                    bm_tier_o   = bm_tier[other_reco_mask]
                    bm_parent_o = bm_parent[other_reco_mask]
                    mc_parent_o = mc_parent[other_reco_mask]
                
                    if (tier == 0) :
                        n_not_best_match = 0.0
                        n_false_primary  = 0.0
                        n_correct_parent = ak.sum(primary_reco_mask)
                        n_false_parent = ak.sum(other_reco_mask)
                    else :
                        n_not_best_match = ak.sum(bm_parent_o == -1)
                        n_false_primary  = ak.sum(primary_reco_mask)
                        n_correct_parent = ak.sum(bm_parent_o == mc_parent_o)
                        n_false_parent = ak.sum((bm_parent_o != -1) & (bm_parent_o != mc_parent_o))
                
                    frac_not_best_match = round(0.0 if n_other == 0 else float(n_not_best_match) / float(n_other), 2)
                    frac_false_primary = round(0.0 if n_other == 0 else float(n_false_primary) / float(n_other), 2)
                    frac_correct_parent = round(0.0 if n_other == 0 else float(n_correct_parent) / float(n_other), 2)
                    frac_false_parent = round(0.0 if n_other == 0 else float(n_false_parent) / float(n_other), 2)
                
                    print(' ' + str(Definitions.tier_strings[tier]) + str(' '* (10 - len(str(Definitions.tier_strings[tier])))) +
                                                            '|' + str(frac_correct_parent) + str(' '* (16 - len(str(frac_correct_parent)))) + \
                                                            '|' + str(frac_false_primary) + str(' '* (15 - len(str(frac_false_primary)))) + \
                                                            '|' + str(frac_false_parent) + str(' '* (14 - len(str(frac_false_parent)))) + \
                                                            '|' + str(frac_not_best_match) + str(' '* (23 - len(str(frac_not_best_match)))) + \
                                                            '|', file=f)
                print('------------------------------------------------------------------------------------', file=f)
                print('', file=f)
