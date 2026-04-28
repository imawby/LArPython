#####################################################################################################################
#####################################################################################################################

def IsCCNueSignal(nusel_branches, isNu) :
    is_in_fv_mask = IsInFiducialVolume(nusel_branches, True)
    return is_in_fv_mask & (nusel_branches['NuPdg'] == 12 if isNu else nusel_branches['NuPdg'] == -12) & (nusel_branches['NC'] == 0) & (nusel_branches['TargetZ'] == 18)

#####################################################################################################################
#####################################################################################################################

def IsCCNueFlavourSignal(nusel_branches) :
    is_in_fv_mask = IsInFiducialVolume(nusel_branches, True)
    return is_in_fv_mask & (abs(nusel_branches['NuPdg']) == 12) & (nusel_branches['NC'] == 0) & (nusel_branches['TargetZ'] == 18)

#####################################################################################################################
#####################################################################################################################

def IsCCNueFlavourOutOfFV(nusel_branches) :
    is_in_fv_mask = IsInFiducialVolume(nusel_branches, True)
    return (~is_in_fv_mask) & (abs(nusel_branches['NuPdg']) == 12) & (nusel_branches['NC'] == 0) 

#####################################################################################################################
#####################################################################################################################

def IsCCNumuSignal(nusel_branches, isNu) :
    is_in_fv_mask = IsInFiducialVolume(nusel_branches, True)
    return is_in_fv_mask & (nusel_branches['NuPdg'] == 14 if isNu else nusel_branches['NuPdg'] == -14) & (nusel_branches['NC'] == 0) & (nusel_branches['TargetZ'] == 18)

#####################################################################################################################
#####################################################################################################################

def IsCCNumuFlavourSignal(nusel_branches) :
    is_in_fv_mask = IsInFiducialVolume(nusel_branches, True)
    return is_in_fv_mask & (abs(nusel_branches['NuPdg']) == 14) & (nusel_branches['NC'] == 0) & (nusel_branches['TargetZ'] == 18)

#####################################################################################################################
#####################################################################################################################

def IsCCNumuFlavourOutOfFV(nusel_branches) :
    is_in_fv_mask = IsInFiducialVolume(nusel_branches, True)
    return (~is_in_fv_mask) & (abs(nusel_branches['NuPdg']) == 14) & (nusel_branches['NC'] == 0)     

#####################################################################################################################
#####################################################################################################################

def IsCCNutauFlavour(nusel_branches) :
    return (abs(nusel_branches['NuPdg']) == 16) & (nusel_branches['NC'] == 0)

#####################################################################################################################
#####################################################################################################################

def IsNC(nusel_branches) :
    return (nusel_branches['NC'] == 1)    

#####################################################################################################################
#####################################################################################################################

def IsInFiducialVolume(nusel_branches, isTrue) :
    minX, maxX = -360.0 + 50.0, 360.0 - 50.0
    minY, maxY = -600.0 + 50.0, 600.0 - 50.0
    minZ, maxZ = 50.0, 1394.0 - 150.0

    true_x = nusel_branches['NuX'] if isTrue else nusel_branches['RecoNuVtxX']
    true_y = nusel_branches['NuY'] if isTrue else nusel_branches['RecoNuVtxY']
    true_z = nusel_branches['NuZ'] if isTrue else nusel_branches['RecoNuVtxZ']

    is_in_fv_mask = (true_x > minX) & (true_x < maxX) &\
                    (true_y > minY) & (true_y < maxY) &\
                    (true_z > minZ) & (true_z < maxZ)

    return is_in_fv_mask;

#####################################################################################################################
#####################################################################################################################



