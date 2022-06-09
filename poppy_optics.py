import numpy as np
import astropy.units as u
from astropy.io import fits
import os
from pathlib import Path

import poppy
from poppy.poppy_core import PlaneType

import cupy as cp

import misc

dm_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data/dm-acts')

npix = 256
oversample = 4
wavelength = 500e-9*u.m

# HST example - Following example in PROPER Manual V2.0 page 49.
# This is an idealized case and does not correspond precisely to the real telescope
diam = 2.4 * u.m
fl_pri = 5.52085 * u.m
d_pri_sec = 4.907028205 * u.m
fl_sec = -0.6790325 * u.m
d_sec_to_focus = 6.3919974 * u.m

m1 = poppy.QuadraticLens(fl_pri, name='Primary')
m2 = poppy.QuadraticLens(fl_sec, name='Secondary')
angular_focal_plane = poppy.ScalarTransmission(planetype=PlaneType.image, name='Focus')

# Pupil optics
circ_pupil = poppy.CircularAperture(radius=diam/2)
multi_circ_pupil = poppy.MultiCircularAperture(rings=1, segment_radius=diam/8, gap = 0.1,)
rect_pupil = poppy.RectangleAperture(width=diam, height=diam/2, rotation=10,)
square_pupil = poppy.SquareAperture(size=diam, rotation=90)
hex_pupil_side = poppy.HexagonAperture(side=diam/2, rotation=0*u.degree)
hex_pupil_flat = poppy.HexagonAperture(flattoflat=diam*np.sqrt(2)/2, rotation=0)
multi_hex_pupil_side = poppy.MultiHexagonAperture(rings=3, side=0.2*u.m, rotation=90) 
multi_hex_pupil_flat = poppy.MultiHexagonAperture(rings=3, flattoflat=0.4*u.m, rotation=90) 
ngon_pupil = poppy.NgonAperture(nsides=5, radius=diam/2, rotation=0.,)
sec_obs = poppy.SecondaryObscuration(secondary_radius=0.396, support_width=0.0264, support_angle_offset=45.0)
asym_sec_obs = poppy.AsymmetricSecondaryObscuration(secondary_radius=diam/4, support_angle=(60, 180, 300), 
                                                    support_width=0.1*u.m, support_offset_x=0.0, support_offset_y=0.0,)
compound_pupil = poppy.CompoundAnalyticOptic(opticslist=[circ_pupil, asym_sec_obs])

# Misc analytic optics
thin_lens = poppy.ThinLens(nwaves=1.0, reference_wavelength=wavelength, radius=diam/2,)
gauss_ap = poppy.GaussianAperture(fwhm=diam/2, w=None, pupil_diam=diam)
knife_edge = poppy.KnifeEdge(shift_x=None, rotation=45, planetype=PlaneType.intermediate)
inv_circ = poppy.InverseTransmission(poppy.CircularAperture(radius=diam/8))
scalar_trans_pupil = poppy.ScalarTransmission(planetype=PlaneType.pupil)
scalar_trans_image = poppy.ScalarTransmission(planetype=PlaneType.image)
scalar_trans_int = poppy.ScalarTransmission(planetype=PlaneType.intermediate)
scalar_opd = poppy.ScalarOpticalPathDifference(opd=499e-9)

# WFE optics
zern_wfe = poppy.ZernikeWFE(coefficients=[0,0,0,0,0,0,0,0,wavelength.value/40], radius=diam/2, aperture_stop=True)
sin_wfe = poppy.SineWaveWFE(spatialfreq=3.0, amplitude=0.5e-6, phaseoffset=0, rotation=90)
kol_wfe = poppy.KolmogorovWFE(r0=diam, dz=10000*u.mm, seed=123456)
psd_wfe = poppy.StatisticalPSDWFE(index=3.0, wfe=wavelength/5, radius=1*u.meter, seed=123456,)

# Power Spectrum parameters
space_unit = u.m # POPPY defaults to meters, this is mandatory.
surface_unit = u.nm # Adjustable with modeling. PowerSpectrumWFE will produce opd in meters as POPPY requires.
alpha = 1.55
beta = 0.637 * (surface_unit**2) / (space_unit**(alpha-2))
oscl = (0.1*space_unit)
iscl = 0.0003
bsr = 1e-8 * (surface_unit*space_unit)**2 # original test case value is 0, adjusted for demo
psd_params = [[alpha, beta, oscl, iscl, bsr]]
weight = [1.0]
power_spec_wfe = poppy.PowerSpectrumWFE(psd_parameters=psd_params, psd_weight=weight, seed=123456, apply_reflection=True, 
                                        screen_size=round(npix*oversample), rms=None, incident_angle=0*u.deg, radius=None, )


# Array and FITS optics
roman_pupil = fits.getdata('fits-files/roman_pupil_309.fits')
roman_primary_opd = fits.getdata('fits-files/roman_primary_opd_309.fits')
dm_data = fits.getdata('fits-files/roman_hlc_dm1.fits')

if poppy.accel_math._USE_CUPY:
    rpupil_array = poppy.ArrayOpticalElement(transmission=cp.array(roman_pupil), pixelscale=diam/310/u.pix)
    rprimary_opd_array = poppy.ArrayOpticalElement(opd=cp.array(roman_primary_opd),
                                                   pixelscale=diam/310/u.pix, planetype=PlaneType.pupil)
    rpupil_opd_array = poppy.ArrayOpticalElement(transmission=cp.array(roman_pupil), opd=cp.array(roman_primary_opd), 
                                                 pixelscale=diam/310/u.pix)
else:
    rpupil_array = poppy.ArrayOpticalElement(transmission=np.array(roman_pupil), pixelscale=diam/310/u.pix)
    rprimary_opd_array = poppy.ArrayOpticalElement(opd=np.array(roman_primary_opd),
                                                   pixelscale=diam/310/u.pix, planetype=PlaneType.pupil)
    rpupil_opd_array = poppy.ArrayOpticalElement(transmission=np.array(roman_pupil), opd=np.array(roman_primary_opd), 
                                                 pixelscale=diam/310/u.pix)
rpupil = poppy.FITSOpticalElement('Roman Pupil', transmission='fits-files/roman_pupil_309.fits',
                                  pixelscale=diam.value/310, planetype=PlaneType.pupil)
rprimary_opd = poppy.FITSOpticalElement('Roman Primary OPD', opd='fits-files/roman_primary_opd_529.fits', opdunits='meters', 
                                        pixelscale=diam.value/529, planetype=PlaneType.pupil)
rpupil_opd = poppy.FITSOpticalElement('Roman Primary', transmission='fits-files/roman_pupil_309.fits',
                                      opd='fits-files/roman_primary_opd_309.fits', opdunits='meters', 
                                      pixelscale=diam.value/310, planetype=PlaneType.pupil)

# Focal plane elements
blc_circ = poppy.BandLimitedCoronagraph(kind='circular', sigma=4, wavelength=None,)
blc_lin = poppy.BandLimitedCoronagraph(kind='linear', sigma=1, wavelength=wavelength,)
fqpm = poppy.IdealFQPM(wavelength=wavelength,)
rect_stop = poppy.RectangularFieldStop(width=0.5*u.arcsec, height=0.2*u.arcsec)
square_stop = poppy.SquareFieldStop(size=0.2*u.arcsec)
annular_stop = poppy.AnnularFieldStop(radius_inner=0.05, radius_outer=0.5)
hex_stop = poppy.HexagonFieldStop(side=0.1*u.arcsec)
circ_occ = poppy.CircularOcculter(radius=0.2)
bar_occ = poppy.BarOcculter(width=0.1, height=1)

osys = poppy.OpticalSystem(pupil_diameter=diam, npix=npix, oversample=oversample)
fosys = poppy.FresnelOpticalSystem(pupil_diameter=diam, npix=npix, beam_ratio=1/oversample)

# Continuous DM
Nact = 48
dm_diam = 46.3*u.mm
act_spacing = 0.9906*u.mm
DM = poppy.ContinuousDeformableMirror(name='DM', dm_shape=(Nact,Nact), actuator_spacing=act_spacing, 
                                      radius=dm_diam/2, influence_func=str(dm_dir/'proper_inf_func.fits'))
dm_data = fits.getdata('fits-files/roman_hlc_dm1.fits')
DM.set_surface(dm_data)

circ_seg_DM = poppy.CircularSegmentedDeformableMirror(rings=1, segment_radius=dm_diam/2/3 - 0.001*u.m/2, gap=0.001*u.m)
hex_seg_DM = poppy.HexSegmentedDeformableMirror(rings=1, flattoflat=dm_diam/3 - 0.001*u.m, gap=0.001 * u.m)

fl_relay = (dm_diam/2) * 1/4
relay = poppy.QuadraticLens(name='Relay to/from Pupil')


# Wavefront for testing
wf = poppy.Wavefront(wavelength=wavelength, diam=diam, npix=npix, oversample=oversample)
wf_dm = poppy.Wavefront(wavelength=wavelength, diam=dm_diam, npix=npix, oversample=oversample)

fwf = poppy.FresnelWavefront(wavelength=wavelength, beam_radius=diam/2, npix=npix, oversample=oversample)
fwf_dm = poppy.FresnelWavefront(wavelength=wavelength, beam_radius=dm_diam/2, npix=npix, oversample=oversample)





