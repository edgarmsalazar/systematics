# ================================================================================================
#
#   This program works with data from a galaxy catalog and handles the information to generate
#   maps. Then extracts systematics maps from another calatog and matches the current footprint.
#   Later, runs a Monte Carlo Markov Chain (MCMC) to apply a log-likelihood model and find a fit
#   for the parameters in the model.
#
#   Outputs:
#       *.txt files for maps
#		*.npy files for maps (faster loading using numpy)
#       *.png or *.pdf plots of the maps
#       *.tex table with MLE results from MCMC
#
# ================================================================================================
#
#   User defined variables are formated as: NameWithNoSpaces
#   User defined functions are formated as: Name_With_Underscores
#
# ================================= PACKAGES AND FUNCTIONS =======================================

from subroutines import *

# =================================== MAIN =======================================================

def main():
    start = time.clock()
    print'# ========================================================='
    print"""    Main program initiated"""
    
    # Define global parameters
    NSIDE      = 64
    
    path       = '/home/edgar/Documents/GSRP/Cluster-Clustering/'
    PixelsPath = path+'class/maps/rasdecs_pixels.txt'
    MaskPath   = path+'catalogs/dr8_depth_i_cmodel_v2.0.fit'
    SysPath    = path+'catalogs/dr8_systematics_2048-16_structure.fit'
    MapsPath   = path+'class/maps/'
    PNGPath    = path+'class/PNG/'
    PDFPath    = path+'class/PDF/'
    
    # Handle Footprint Maps ======================================================================
    
    GalaxyCount, FractionGood, Density, DensityContrast, DensityMean = Generate_Maps(NSIDE, PixelsPath, MaskPath, MapsPath)
    
    # Plot maps
    RAS, DECS = np.loadtxt(MapsPath+'footprint.txt', skiprows = 1, unpack = True)
    Plot_Catalog(RAS, DECS, GalaxyCount, FractionGood, Density, DensityContrast, PNGPath)
    Plot_Catalog(RAS, DECS, GalaxyCount, FractionGood, Density, DensityContrast, PDFPath, 'pdf', 600)
    
    # Handle Systematic Maps =====================================================================

    SystematicsContrast, NCols = Generate_Systematics(NSIDE, DensityContrast, SysPath, MapsPath)
    
    #Plot systematics
    Plot_Systematics(SystematicsContrast, PNGPath)
    Plot_Systematics(SystematicsContrast, PDFPath, 'pdf', 600)
    
    # MCMC =======================================================================================
    
    Params = []
    Params.append("$\sigma^2$")
    for i in range(NCols):
        Params.append("$a_{%s}$" % (i+1))
    Guess = np.zeros(NCols+1)
    Guess[0] = 0.5
           
    Epsilon, DensityContrast, NPoints = MC_MC(NSIDE, GalaxyCount, FractionGood, DensityContrast, 
										DensityMean, SystematicsContrast, Guess, Params, MapsPath,
										Nsteps = 20000, Factor = 2)
    
    DensityContrast = np.load(MapsPath+'npy/DensityContrast.npy')
    Epsilon         = np.load(MapsPath+'npy/Epsilon.npy')
    
    NCols  = Epsilon.shape[1]
    Params = []
    Params.append("$\sigma$")
    for i in range(NCols):
        Params.append("$a_{%s}$" % (i+1))
    
    MLE = [0.1637, -0.007, -0.190, 0.062, -0.030, 0.063, 0.143, -0.040, 
		   0.049, -0.068, -0.003, -0.062, -0.150, -0.780, -0.118, -0.101]
    
    Plot_Chain(DensityContrast, Epsilon, MLE, Params, PNGPath, PDFPath)

    finish = time.clock()
    Run_Time(start,finish,'Main program done in')
    print'# ========================================================='
    
    return
    
if __name__ == "__main__":
    main()
