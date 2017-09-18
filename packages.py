# ================================================================================================
#
#   This file contains all packages required by the main file. To use file just add the next line
#   at the begging of the main program.
#
#   from packages import *
#
# ================================= PACKAGES =====================================================
import numpy as np
import healpy as hp
import time
import emcee
from astropy.io import fits
from chainconsumer import ChainConsumer

# To make plots
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
#import matplotlib
#matplotlib.rcParams.update({'font.size': 16})
