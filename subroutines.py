# ================================================================================================
#
#   This file contains callable subroutines to process data. To se file just add the next line at
#   the begginig of the main program.
#
#	from subroutines import *
#
# ================================================================================================
#
#   User defined variables are formated as: NameWithNoSpaces
#   User defined functions are formated as: Name_With_Underscores
#
# ================================= PACKAGES AND FUNCTIONS =======================================

from functions import *

# ================================= SUBROUTINES ==================================================

def Generate_Maps(NSIDE,PixelsPath, MaskPath, MapsPath):
    start = time.clock()
    print'# ========================================================='
    print"""    Generate Maps initiated"""
    
    # Get number of pixels in the map
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    
    # Part 1: Get the catalog footprint ==========================================================
    # Processed data from Calvin due to package issues.
    Pixels = np.loadtxt(PixelsPath, dtype = int, skiprows = 1, unpack = True)
    
    # Part 2: Get the raw galaxy map =============================================================
    # Count the number of objects in each pixel
    GalaxyCount = Bin_Data(Pixels, NPIX)
    print"""    Galaxy map (RAW) done."""
    
    # Part 3: Get mask ===========================================================================
    hduMask      = fits.getdata(MaskPath, ext=1, header=True)
    NSIDEMask    = hduMask[1]['NSIDE']
    MaskPixels   = hduMask[0]['HPIX']
    FractionGood = hduMask[0]['FRACGOOD']

    # Convert the frac good data to a mask
    Mask = Convert_to_Map(MaskPixels, FractionGood, hp.nside2npix(NSIDEMask))
    print"""    Mask (RAW) done."""
    
    # Part 4: Apply the mask and generate density maps ===========================================
    if not NSIDE == NSIDEMask:
        print"""    Updating mask resolution to map resolution."""
        Mask = hp.ud_grade(Mask, NSIDE)
        
    AreaPixel = hp.pixelfunc.nside2pixarea(NSIDE,degrees=True)
    Density   = True_Divide(GalaxyCount, Mask) / AreaPixel
    print"""    Masking done."""
    
    DensityMean = Get_Mean(Density)
    DensityContrast = Contrast_Map(Density)
    print"""    Density contrast done."""
    
    # Part 5: Save maps ==========================================================================
    txt = 'txt/'
    npy = 'npy/'
    
    Save_Txt(MapsPath+txt+'GalaxyCount.txt', GalaxyCount, "Number of galaxies in a pixel (RAW)")
    Save_Txt(MapsPath+txt+'FractionGood.txt', Mask, "Frac_good mask with updated reslution")
    Save_Txt(MapsPath+txt+'Density.txt', Density, "Density (galaxies per pixel)")
    Save_Txt(MapsPath+txt+'DensityContrast.txt', DensityContrast, "Density contrast \Delta")
    np.save(MapsPath+npy+'GalaxyCount', GalaxyCount)
    np.save(MapsPath+npy+'FractionGood', Mask)
    np.save(MapsPath+npy+'Density', Density)
    np.save(MapsPath+npy+'DensityContrast', DensityContrast)
    print"""    Saving maps done."""
    
    # ============================================================================================
    # Final Messages
    print"""
    Using nside              = {0}
    Total number of pixels   = {1}
    Area per pixel           = {2} deg2
    Average galaxy           = {3} galaxies / pixel
    Average density          = {4} galaxies / deg2
    Average density contrast = {5}
    """.format(NSIDE,NPIX,AreaPixel,Get_Mean(GalaxyCount),DensityMean,Get_Mean(DensityContrast,-1.))   
    
    finish = time.clock()
    Run_Time(start,finish,'Generate Maps finished in')
    print'# ========================================================='
    
    return GalaxyCount, Mask, Density, DensityContrast, DensityMean

def Generate_Systematics(NSIDE, ReferenceMap, SysPath, MapsPath):
    start = time.clock()
    print'# ========================================================='
    print"""    Generate Systematics Maps initiated"""

    # Get number of pixels in the map
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    
    # Part 1: Load maps ==========================================================================
    # Load in the table from the catalog
    Hdu          = fits.getdata(SysPath, ext=1, header=True)
    NSIDE_Hdu    = fits.open(SysPath)[1].header['NSIDE']
    NPIX_Hdu     = hp.nside2npix(NSIDE_Hdu)
    Hpix         = np.array(Hdu[0]['HPIX'])
    FractionGood = np.array(Hdu[0]['FRACGOOD'])
    PSFAll       = np.array(Hdu[0]['PSF_FWHM'])
    SkyFluxAll   = np.array(Hdu[0]['SKYFLUX'])
    SkySigAll    = np.array(Hdu[0]['SKYSIG'])
    AirMass      = np.array(Hdu[0]['AIRMASS'])
    
    # Stack all data into one giant array
    SystematicsArray = np.vstack((PSFAll.T,SkyFluxAll.T,SkySigAll.T,AirMass.T)).T
    # Number of columns in array
    NCols =  SystematicsArray.shape[1]
    print"""    Loading catalog done."""
    
    print"""
    NSIDE        = {0}
    NPIX         = {1}
    NCols (maps) = {2}
    """.format(NSIDE_Hdu,NPIX_Hdu, NCols)
    
    # Part 2: Convert to map =====================================================================
    Systematics = np.empty([NPIX_Hdu,NCols])
    FractionGood = Convert_to_Map(Hpix,FractionGood,NPIX_Hdu)
    for i in range(NCols):
        Systematics[:,i] = Convert_to_Map(Hpix, SystematicsArray[:,i], NPIX_Hdu)
    print"""    Conversion done."""
    
    # Convert maps from Nested ordering to Ring ordering (easy to handle)
    FractionGood = hp.pixelfunc.reorder(FractionGood, inp = 'NESTED', out = 'RING')
    for i in range(NCols):
        Systematics[:,i] = hp.pixelfunc.reorder(Systematics[:,i], inp = 'NESTED', out = 'RING')
    print"""    Reordering done."""
    
    # Part 3: Properly update resolution to match the reference map ==============================
    # Update maps resolution
    if not NSIDE == NSIDE_Hdu:
        print"""    Updating mask resolution to map resolution."""
        SystematicsMask = np.empty([NPIX_Hdu,NCols])
        SystematicsUpdate = np.empty([NPIX,NCols])
        
        # Mask frac_good into systematic maps
        for i in range(NCols):
            SystematicsMask[:,i] = Systematics[:,i] * FractionGood
            
        # Update resolution for FracGood and then for all systematics
        FractionGood = hp.ud_grade(FractionGood, NSIDE)
        for i in range(NCols): 
            SystematicsUpdate[:,i] = hp.ud_grade(SystematicsMask[:,i], NSIDE)
        
        # Delete array 'Systematics' and create another with the new dimensions
        del Systematics
        Systematics = np.empty([NPIX,NCols])
        for i in range(NCols):
            Systematics[:,i] = True_Divide(SystematicsUpdate[:,i],FractionGood)
    
    # Compare to reference map (density contrast)
    for i in range(NCols):
        Systematics[:,i] = Footprint_Match(ReferenceMap, Systematics[:,i],-1.)
    print"""    Map processing done."""
    
    # Part 4: Generate contrast maps =============================================================
    for i in range(NCols):
        Systematics[:,i] = Contrast_Map(Systematics[:,i])
    
    # Save maps    
    txt = 'txt/'
    npy = 'npy/'
    Save_Txt(MapsPath+txt+'SystematicsContrast.txt', Systematics, "Systematics Contrast Maps")
    np.save(MapsPath+npy+'SystematicsContrast', Systematics)
    print"""    Constast maps done."""
    
    finish = time.clock()
    Run_Time(start,finish,'Sytematics Maps Processing finished in')
    print'# ========================================================='
    
    return Systematics, NCols
    
def MC_MC(NSIDE, GalaxyCount, FractionGood, DensityContrast, DensityMean, Systematics, Guess, Params, MapsPath, Nsteps=10000, Factor=10):
    start = time.clock()
    print'# ========================================================='
    print"""    MCMC processing initiated for {} parameters""".format(len(Guess))
    
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    GalaxyMean = DensityMean * hp.pixelfunc.nside2pixarea(NSIDE,degrees=True)
    
    # Part 1: Cut maps ===========================================================================
    # Get just the pixels with data (those are the only ones that really matter)
    NCols, m = Systematics.shape[1], 0
    for i in range(NPIX):
        if np.isclose(GalaxyCount[i],0.)==False:m+=1
    Epsilon = np.empty([m,NCols])
    for i in range(NCols):
        Epsilon[:,i] = Map_Cut(GalaxyCount,Systematics[:,i])
    FractionGood     = Map_Cut(GalaxyCount,FractionGood)
    DensityContrast  = Map_Cut(GalaxyCount,DensityContrast)
    GalaxyCount      = Map_Cut(GalaxyCount,GalaxyCount)
    
    print"""
    Number of relevant pixels out of {0}
    Epsilon      = {1}
    FractionGood = {2}
    GalaxyCount  = {3}
    """.format(NPIX, Epsilon.shape, FractionGood.shape, GalaxyCount.shape)
    
    # Save cut maps
    txt = 'txt/'
    npy = 'npy/'
    Save_Txt(MapsPath+txt+'Epsilon.txt',Epsilon,"Systematic Contrast Maps with only relevant pixels")
    np.save(MapsPath+npy+'Epsilon',Epsilon)
    
    print"""    Eliminating empty pixels done."""
    
    # Part 2: Set up sampler =====================================================================
    Ndim = len(Guess)
    Nwalkers = Factor*Ndim
    Position = [Guess + 1e-2*np.random.randn(Ndim) for i in range(Nwalkers)]
    
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnlike, args=(GalaxyCount, FractionGood, GalaxyMean, Epsilon))
    
    print"""    Sampler set.
    
    Initial guess = {0}
    Dimensions    = {1}
    Walkers       = {2}
    Steps         = {3}
    """.format(Guess, Ndim, Nwalkers, Nsteps)
    
    # Run MCMC
    print"""    Running MCMC."""
    sampler.run_mcmc(Position, Nsteps)
    Burning = int(0.2*Nsteps)
    Samples = sampler.chain[:,Burning:,:].reshape((-1, Ndim))
    np.save('chain',Samples)
    
    # Part 3: Plot Results =======================================================================
    # Plot walks
    c = ChainConsumer()
    c.add_chain(Samples, parameters=Params)
    fig_walks = c.plotter.plot_walks()
    plt.savefig('mcmc_walks.png')
    plt.clf()
    
    # Plot posteriors and contours
    c = ChainConsumer()
    c.add_chain(Samples, parameters=Params, name='MLE')
    c.configure(diagonal_tick_labels=False, tick_font_size=6, label_font_size=10, max_ticks=6, colors="#388E3C")
    fig = c.plotter.plot(filename="mcmc.jpg", figsize="page", parameters=Params[:])
    #fig2 = c.plotter.plot(filename="mcmc.pdf", figsize="page", parameters=Params[:])
    plt.clf()
    
    # Export results
    table = c.analysis.get_latex_table(caption="Results for the tested models", label="tab:example")
    f = open( 'mcmc_MLE.tex', 'w' )
    f.write( table )
    f.close()
    
    finish = time.clock()
    Run_Time(start,finish,'MCMC done in')   
    print'# ========================================================='
    
    return Epsilon, DensityContrast, len(DensityContrast)

def Plot_Catalog(RAS, DECS, GalaxyCount, Mask, Density, DensityContrast, path, extension='png',dpi=300):
    start = time.clock()
    print'# ========================================================='
    print"""    Plot Catalog Maps initiated"""

    # PNG output
    if extension=='png':
        plt.figure(figsize = (8,4.5))
        plt.xlabel(r'Right Asension (deg)')
        plt.ylabel(r'Declination (deg)')
        plt.title('Catalog Footprint')
        plt.plot(RAS, DECS, '.', c = 'xkcd:tomato')
        plt.savefig(path+'001'+'.'+extension,dpi=dpi)
        plt.clf()
    
    hp.mollview(GalaxyCount, title='Galaxy Map ($\\vec{N}^{raw})$ ', unit=r'galaxies')
    plt.savefig(path+'002'+'.'+extension,dpi=dpi)
    plt.clf()
    
    hp.mollview(Mask, title='Updated Mask ($\\vec{f}$)', unit = 'Fraction good', max = 1, min = 0)
    plt.savefig(path+'003'+'.'+extension,dpi=dpi)
    plt.clf()

    hp.mollview(Density, title='Galaxy Density ($\\vec{n}$)', unit = 'galaxies/deg$^{2}$', max = 120, min = 0)
    plt.savefig(path+'004'+'.'+extension,dpi=dpi)
    plt.clf()

    hp.mollview(DensityContrast, title='Density Contrast Map ($\\vec{\Delta}$)', min=-1, max=1)
    plt.savefig(path+'005'+'.'+extension,dpi=dpi)
    plt.clf()
    
    print"""    Plotting done with extension {}.""".format(extension)
    
    finish = time.clock()
    Run_Time(start,finish,'Plot Catalog Maps finished in')
    print'# ========================================================='
    
    return

def Plot_Systematics(SystematicsContrast, path, extension='png',dpi=300):
    start = time.clock()
    print'# ========================================================='
    print"""    Plot Systematic Maps initiated"""
    
    for i in range(SystematicsContrast.shape[1]):
        hp.mollview(SystematicsContrast[:,i], title='Systematic Map {}'.format(i+1))
        plt.savefig(path+'sys{}'.format(i+1)+'.'+extension,dpi=dpi)
        plt.clf()

    print"""    Plotting done with extension {}.""".format(extension)    
    
    finish = time.clock()
    Run_Time(start,finish,'Sytematics Maps Processing finished in')
    print'# ========================================================='
    
    return

def Plot_Chain(ObservedDensityContrast, Epsilon, MLE, Params, PNGPath, PDFPath):
    start = time.clock()
    print'# ========================================================='
    ObservedDensityContrast = Map_Cut(ObservedDensityContrast,ObservedDensityContrast, -1.)
    np.save('DensityContrastCut',ObservedDensityContrast)
    NPoints = len(ObservedDensityContrast)
    
    print"""    Chain Plot initiated for {0} pixels out of {1}""".format(NPoints,49152)
    dpi = 300
    std = np.sqrt(MLE[0])
    weights  = MLE[1:]

    # Part 1: Load Chain =========================================================================
    Chain = np.load('chain.npy')
    c = ChainConsumer()
    c.add_chain(Chain, parameters=Params, name='MLE')
    c.configure(plot_hists=False, diagonal_tick_labels=True, tick_font_size=6, label_font_size=10, max_ticks=5, colors="#388E3C")
    fig = c.plotter.plot(filename=PNGPath+"Countours.png", figsize="page", parameters=Params[:])
    fig2= c.plotter.plot(filename=PDFPath+"Countours.pdf", figsize="page", parameters=Params[:])
    plt.clf()
    
    # Part 2: Recover underlying density contrast ================================================
    
    zeros = np.zeros(NPoints)
    for i in range(len(weights)):
        zeros += weights[i]*Epsilon[:,i]
    
    NewDC = ObservedDensityContrast - zeros
    
    x = np.linspace(-5*np.sqrt(np.var(NewDC)), 5*np.sqrt(np.var(NewDC)), 400)
    bins = 200
    
    plt.hist(ObservedDensityContrast, bins, range=(min(x), max(x)), normed = True, color = 'forestgreen', alpha=0.5, label = "$\Delta$",zorder=0)
    plt.hist(NewDC, bins, range=(min(x), max(x)), normed = True, color = 'royalblue', alpha=0.5, label = "$\Delta-\\sum a_\\alpha \epsilon_\\alpha$",zorder=1)
    plt.plot(x,mlab.normpdf(x, 0., std), color='red', label = "$N({0},{1:.4f})$".format(0.,std), zorder = 2)
    #plt.plot(x,mlab.normpdf(x, 0., std**2), color='red', label = "$N({0},{1:.4f})$".format(0.,std**2), zorder = 3)
    plt.plot(x,mlab.normpdf(x, 0., np.sqrt(np.var(NewDC))), color='yellow', label = "$N({0},{1:.4f})$".format(0.,np.sqrt(np.var(NewDC))), zorder = 4)
    plt.title("Density Contrast")
    plt.legend(loc='best')
    plt.savefig(PNGPath+'Contrast.png',dpi=dpi)
    plt.savefig(PDFPath+'Contrast.pdf',dpi=dpi)
    plt.clf()

    finish = time.clock()
    Run_Time(start,finish,'Plotting done in')   
    print'# ========================================================='
    
    return
