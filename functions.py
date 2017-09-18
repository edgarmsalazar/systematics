# ================================================================================================
#
#   This file contains all functions required by the main file. To use file just add the next line
#   at the begging of the main program.
#
#   from functions import *
#
# ================================================================================================
#
#   User defined variables are formated as: NameWithNoSpaces
#   User defined functions are formated as: Name_With_Underscores
#
# ================================= PACKAGES =====================================================

from packages import *

# ================================= FUNCTIONS ====================================================
def Run_Time(Start,Finish,message='Program done in'):
    TotalTime = Finish-Start
    if (TotalTime > 3600.):
        Hours = TotalTime / 3600.
        Minutes = (Hours - int(Hours))*60.
        Seconds = (Minutes - int(Minutes))*60.
        print"""    """+message+""" %.i : %.i : %.5f hours"""%(Hours,Minutes,Seconds)
    elif (TotalTime > 60.):
        Minutes = TotalTime / 60.
        Seconds = (Minutes - int(Minutes))*60.
        print"""    """+message+""" %.i : %.5f minutes"""%(Minutes,Seconds)
    else:
        Seconds = TotalTime
        print"""    """+message+""" %.5f seconds"""%(Seconds)
    return

def Save_Txt(FileName,Data,Header):
    with open(FileName, "w") as output:
        output.writelines('# === '+Header+' ====\n')
        for i in range(len(Data)): output.writelines("%s \n" % Data[i])
    return

# Bypass division by 0.
def True_Divide(Num, Den):
    with np.errstate(divide='ignore', invalid='ignore'):
        Cocient = np.divide(Num,Den)
        Cocient[ ~ np.isfinite(Cocient)] = 0.0
    return Cocient

# Pixels with no relevant data are excluded from calculation.
def Get_Mean(Data,Val=0.):
    n, m, Mean = len(Data), 0. ,0.
    for i in range(n):
        if (np.isclose(Data[i],Val) == False):
            Mean += Data[i]
            m += 1
    return Mean / m

#For Galaxy Count Maps
def Bin_Data(Pixels,NPIX):
    BinnedPixels = np.zeros(NPIX, dtype=int)
    for i in Pixels:
        BinnedPixels[i] = BinnedPixels[i] + 1
    return BinnedPixels

# Takes pixel index and place value
def Convert_to_Map(Pixels, OldMap, NPIX):
    NewMap = np.zeros(NPIX)
    NewMap[Pixels] = OldMap
    return NewMap

# Match Map to a Reference to get exact same footprint.
def Footprint_Match(Ref,Map,Val = 0.):
    NewMap = np.zeros(len(Ref))
    for i in range(len(Ref)):
        if (np.isclose(Ref[i],Val) == False):
            NewMap[i] = Map[i]
    return NewMap

# Remove pixels with no relevant data.
def Map_Cut(Ref,Map,Val=0.):
    NewMap = []
    for i in range(len(Ref)):
        if (np.isclose(Ref[i],Val) == False):
            NewMap.append(Map[i])
    return np.array(NewMap)
    
# Extract a column of data from HDU stakc of same type.
def Extract_Column(Map,Col=0):
    n, Column = len(Map), []
    for i in range(n): Column.append(Map[i][Col])
    return Column

# Generate contrast map
def Contrast_Map(Map,Val=0.):
    NewMap = np.zeros(len(Map))
    Mean   = Get_Mean(Map)
    for i in range(len(Map)):
        if (np.isclose(Map[i],Val) == False):
            NewMap[i] = (Map[i] - Mean) / Mean
        else:
            NewMap[i] = -1.0
    return NewMap

# ================================= MCMC Definitions =============================================
def lnprior(Params):
    sigma = Params[0]
    if sigma < 0. : return -np.inf
    for x in Params[1:]:
        if x < -10 or x > 10: return -np.inf
        
    return 0

def lnprob(Params, ObservedGalaxyCount, FractionGood, GalaxyMean, Epsilon):
    ones = np.ones(len(ObservedGalaxyCount))
    sigma   = Params[0]
    weights = Params[1:]
    for i in range(len(weights)):
        ones += weights[i]*Epsilon[:,i]
    mean = GalaxyMean * FractionGood * ones
    var  = mean + (sigma*mean)**2
    var[np.where(var <= 0.)] = 1e-4
    # Compute function
    lnvar    = np.log(var)
    fracs    = (ObservedGalaxyCount  - mean )**2 / var
    sumation = sum(fracs) + sum(lnvar)
    return -0.5*sumation

def lnlike(Params, ObservedGalaxyCount, FractionGood, GalaxyMean, Epsilon):
    lp = lnprior(Params)
    if not np.isfinite(lp):
        return -np.inf
    lf = lnprob(Params, ObservedGalaxyCount, FractionGood, GalaxyMean, Epsilon)
    if not np.isfinite(lf):
        return -np.inf
    return lp + lf
