"""
Helper functions for the DESI DR1 quasar VAC tutorials.

- read the per-object PyQSOFit output files (op_<OBJID>.fits.gz)
- rebuild the continuum (power law + polynomial + FeII) and Gaussian line models
- recompute the line properties stored in the VAC (peak, flux, logL, FWHM, EW, centroid)
- make the QA plot of a fit
- density/contour helpers for catalog-level plots

The model functions follow the code used to compile the VAC, so the line properties computed here
reproduce the catalog values.
"""

import os
import numpy as np
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate, integrate
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
C_KMS = const.c.to(u.km/u.s).value

_DIR = os.path.dirname(os.path.abspath(__file__))
QSOPAR_FILE = os.path.join(_DIR, 'qsopar.fits')


# ------------------------------------------------------------------------------------------------
# plotting style
# ------------------------------------------------------------------------------------------------
def set_mpl_style(fsize=15, tsize=18, tdir='in', major=8.0, minor=4.0, lwidth=1.2, lhandle=2.0):
    """Function to set MPL style"""
    plt.style.use('default')
    plt.rcParams['text.usetex'] = False
    plt.rcParams["axes.axisbelow"] = False
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['xtick.major.width'] = lwidth
    plt.rcParams['xtick.minor.width'] = lwidth
    plt.rcParams['ytick.major.width'] = lwidth
    plt.rcParams['ytick.minor.width'] = lwidth
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['legend.handlelength'] = lhandle
    return


# ------------------------------------------------------------------------------------------------
# continuum model
# ------------------------------------------------------------------------------------------------
# FeII templates: UV (Vestergaard & Wilkes 2001; Tsuzuki et al. 2006; Salviander et al. 2007) and
# optical (Boroson & Green 1992), as shipped with PyQSOFit
fe_uv = np.genfromtxt(os.path.join(_DIR, 'fe_uv.txt'))
fe_op = np.genfromtxt(os.path.join(_DIR, 'fe_optical.txt'))
wave_Fe_mgii = 10 ** fe_uv[:, 0]
flux_Fe_mgii = fe_uv[:, 1] * 1e15
ind_Fe_balmer = (10 ** fe_op[:, 0] > 3686.) & (10 ** fe_op[:, 0] < 7484.)
wave_Fe_balmer = 10 ** fe_op[ind_Fe_balmer, 0]
flux_Fe_balmer = fe_op[ind_Fe_balmer, 1] * 1e15

# order of the continuum parameters used throughout (names of the columns in extension 1)
CONTI_KEYWORD = ['PL_norm', 'PL_slope', 'conti_a_0', 'conti_a_1', 'conti_a_2',
                 'Fe_uv_norm', 'Fe_uv_FWHM', 'Fe_uv_shift', 'Fe_op_norm', 'Fe_op_FWHM', 'Fe_op_shift']


def PowerLaw(xval, pp):
    """pp = [norm, slope]; f = norm * (lambda/3000)^slope"""
    return pp[0]*(xval/3000.0)**pp[1]


def PolyConti(xval, pp, x0=3000):
    """pp = [a_0, a_1, a_2]; f = sum_i a_i/1e6 * (lambda-3000)^(i+1)"""
    xval2 = xval - x0
    yvals = [(pp[i] / 1e6) * xval2 ** (i + 1) for i in range(len(pp))]
    return np.sum(yvals, axis=0)


def FeII_convolve(flux_Fe, Fe_FWHM):
    "Broaden the FeII template to Fe_FWHM (km/s); 106.3 km/s is the dispersion of the template"
    if Fe_FWHM < 900.0:
        sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
    else:
        sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
    sig_pix = sig_conv / 106.3
    khalfsz = np.round(4 * sig_pix + 1, 0)
    xx = np.arange(0, khalfsz * 2, 1) - khalfsz
    kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
    kernel = kernel / np.sum(kernel)
    return np.convolve(flux_Fe, kernel, 'same')


def FeII_template(wave_Fe, flux_Fe, Fe_FWHM):
    "Broadened FeII template as a function of rest wavelength (cubic spline, as in PyQSOFit)"
    tck = interpolate.splrep(wave_Fe, FeII_convolve(flux_Fe, Fe_FWHM))
    return lambda x: interpolate.splev(x, tck)


def FeII_eval(xval, norm, shift, template, wmin, wmax):
    "Evaluate a broadened FeII template on xval; zero unless > 100 pixels fall in (wmin, wmax)"
    yval = np.zeros_like(xval, dtype=float)
    xval_new = xval * (1.0 + shift)
    ind = (xval_new > wmin) & (xval_new < wmax)
    if np.sum(ind) > 100:
        yval[ind] = norm * template(xval_new[ind])
    return yval


def FeII_uv_Conti(xval, pp):
    "UV FeII on the continuum from 1200 to 3500 A. pp = [norm, FWHM, shift]"
    return FeII_eval(xval, pp[0], pp[2], FeII_template(wave_Fe_mgii, flux_Fe_mgii, pp[1]), 1200., 3500.)


def FeII_opt_Conti(xval, pp):
    "Optical FeII on the continuum from 3686 to 7484 A. pp = [norm, FWHM, shift]"
    return FeII_eval(xval, pp[0], pp[2], FeII_template(wave_Fe_balmer, flux_Fe_balmer, pp[1]), 3686., 7484.)


def continuum_PL_poly(wave_val, conti_para):
    "Power law + polynomial part of the continuum"
    return PowerLaw(wave_val, conti_para[0:2]) + PolyConti(wave_val, conti_para[2:5])


def continuum_all(wave_val, conti_para):
    "Full continuum model: power law + polynomial + UV FeII + optical FeII (conti_para in CONTI_KEYWORD order)"
    return continuum_PL_poly(wave_val, conti_para)\
        + FeII_uv_Conti(wave_val, conti_para[5:8])\
        + FeII_opt_Conti(wave_val, conti_para[8:])


def make_continuum(conti_para):
    """
    Same model as continuum_all, but the FeII templates are broadened only once, so the returned
    function is cheap to call many times with the same continuum parameters.
    """
    conti_para = np.asarray(conti_para, dtype=float)
    tmpl_uv = FeII_template(wave_Fe_mgii, flux_Fe_mgii, conti_para[6])
    tmpl_op = FeII_template(wave_Fe_balmer, flux_Fe_balmer, conti_para[9])

    def conti_func(wave_val):
        return continuum_PL_poly(wave_val, conti_para)\
            + FeII_eval(wave_val, conti_para[5], conti_para[7], tmpl_uv, 1200., 3500.)\
            + FeII_eval(wave_val, conti_para[8], conti_para[10], tmpl_op, 3686., 7484.)
    return conti_func


# ------------------------------------------------------------------------------------------------
# reading the PyQSOFit output
# ------------------------------------------------------------------------------------------------
def read_op(filename):
    """
    Read one PyQSOFit output file.
    Returns (op_data, op_data_MC, op_data_spec) = extensions 1 (best fit), 2 (MC trials), 3 (spectra).
    """
    with fits.open(filename) as hdul:
        op_data = hdul[1].data
        op_data_MC = hdul[2].data
        op_data_spec = hdul[3].data
    return op_data, op_data_MC, op_data_spec


def get_conti_para(op_data):
    "Best-fit continuum parameters, in CONTI_KEYWORD order"
    return np.array([op_data[name][0] for name in CONTI_KEYWORD], dtype=float)


def has_host(op_data_spec):
    "True if a host-galaxy decomposition was performed (z < 1 objects)"
    return 'wave_hostfree' in op_data_spec.names


def get_spectrum(op_data_spec, hostfree=False):
    """
    (wave, flux, err) of the rest-frame, Galactic-dereddened spectrum.
    hostfree=True returns the host-subtracted spectrum, on which the continuum and lines were fitted
    when a host decomposition was performed. The host-free columns are zero-padded to the length of
    the table, so the padding is removed here.
    """
    if hostfree and has_host(op_data_spec):
        n_valid = np.sum(op_data_spec['wave_hostfree'] > 0)
        cols = ['wave_hostfree', 'flux_hostfree', 'err_hostfree']
    else:
        n_valid = np.sum(op_data_spec['wave_prereduced'] > 0)
        cols = ['wave_prereduced', 'flux_prereduced', 'err_prereduced']
    return tuple(np.asarray(op_data_spec[c][:n_valid], dtype=float) for c in cols)


# ------------------------------------------------------------------------------------------------
# Gaussian line model
# ------------------------------------------------------------------------------------------------
def Manygauss(xval, pp):
    """
    Sum of Gaussians in ln(wavelength) space.
    xval: ln(wavelength); pp: (ngauss, 3) array of [scale, ln(centerwave), sigma]
    """
    pp = np.asarray(pp, dtype=float).reshape(-1, 3)
    return np.sum(pp[:, 0] * np.exp(-(xval[:, np.newaxis] - pp[:, 1]) ** 2 / (2 * pp[:, 2] ** 2)), axis=1)


def gauss_names(op_data, local=False):
    """
    Names of the Gaussian components saved in the file (e.g. 'CIV_br_1').
    local=False: components of the global complex fits; local=True: components of the local fits.
    """
    names = [n[:-len('_scale')] for n in op_data.names if n.endswith('_scale')]
    local_comps = [op_data[n][0].strip() for n in op_data.names
                   if n.startswith('loc_') and n.endswith('_complex_name')]
    local_lines = [LOCAL_LINES[k][1] + '_1' for k in LOCAL_LINES if LOCAL_LINES[k][0] in local_comps]
    if local:
        return [n for n in names if n in local_lines]
    return [n for n in names if n not in local_lines]


def get_gauss(op_data, sublines, op_data_MC=None, nmc=0, positive_only=True):
    """
    (n, 3) array of [scale, ln(centerwave), sigma] for the components in sublines that were fitted.
    op_data_MC, nmc: use Monte Carlo trial nmc of extension 2 instead of the best fit.
    positive_only: drop components with zero scale (the VAC line properties keep them; they only
    change the integration grid).
    """
    pp = []
    for s in sublines:
        if op_data_MC is None:
            if s+'_scale' in op_data.names:
                pp.append([op_data[s+'_scale'][0], op_data[s+'_centerwave'][0], op_data[s+'_sigma'][0]])
        elif s+'_scale_MC' in op_data_MC.names:
            pp.append([op_data_MC[s+'_scale_MC'][nmc], op_data_MC[s+'_centerwave_MC'][nmc],
                       op_data_MC[s+'_sigma_MC'][nmc]])
    pp = np.array(pp, dtype=float).reshape(-1, 3)
    return pp[pp[:, 0] > 0] if positive_only else pp


def line_model(op_data, wave, sublines=None):
    "Total Gaussian line model (global complex fits) on the rest wavelength grid wave"
    if sublines is None:
        sublines = gauss_names(op_data)
    pp = get_gauss(op_data, sublines)
    return Manygauss(np.log(wave), pp) if len(pp) > 0 else np.zeros_like(wave)


# ------------------------------------------------------------------------------------------------
# line definitions and line properties
# ------------------------------------------------------------------------------------------------
# VAC line column -> (rest-frame vacuum wavelength [A], Gaussian components summed for the line)
LINE_LIST = {
    'HALPHA':      (6564.61, ['Halpha_br_1', 'Halpha_br_2', 'Halpha_br_3', 'Halpha_na_1']),
    'HALPHA_BR':   (6564.61, ['Halpha_br_1', 'Halpha_br_2', 'Halpha_br_3']),
    'HALPHA_NA':   (6564.61, ['Halpha_na_1']),
    'NII6549':     (6549.85, ['NII6549_1']),
    'NII6585':     (6585.28, ['NII6585_1']),
    'SII6718':     (6718.29, ['SII6718_1']),
    'SII6732':     (6732.67, ['SII6732_1']),
    'HBETA':       (4862.68, ['Hbeta_br_1', 'Hbeta_br_2', 'Hbeta_na_1']),
    'HBETA_BR':    (4862.68, ['Hbeta_br_1', 'Hbeta_br_2']),
    'HBETA_NA':    (4862.68, ['Hbeta_na_1']),
    'HEII4687':    (4687.02, ['HeII4687_br_1', 'HeII4687_na_1']),
    'HEII4687_BR': (4687.02, ['HeII4687_br_1']),
    'OIII5007':    (5008.24, ['OIII5007w_1', 'OIII5007c_1']),
    'OIII5007C':   (5008.24, ['OIII5007c_1']),
    'OIII4960':    (4960.30, ['OIII4959c_1', 'OIII4959w_1']),
    'OIII4960C':   (4960.30, ['OIII4959c_1']),
    'HGAMMA':      (4341.68, ['Hgamma_br_1', 'Hgamma_na_1']),
    'HDELTA':      (4102.89, ['Hdelta_br_1', 'Hdelta_na_1']),
    'MGII':        (2798.75, ['MgII_br_1', 'MgII_br_2', 'MgII_na_1']),
    'MGII_BR':     (2798.75, ['MgII_br_1', 'MgII_br_2']),
    'CIII_BR':     (1908.73, ['CIII_br_1', 'CIII_br_2']),
    'SIIII1892':   (1892.03, ['SiIII1892_1']),
    'ALIII1857':   (1857.40, ['AlIII1857_1']),
    'NIII1750':    (1750.26, ['NIII1750_1']),
    'CIV':         (1549.06, ['CIV_br_1', 'CIV_br_2', 'CIV_br_3']),
    'HEII1640':    (1640.42, ['HeII1640_1', 'HeII1640_br_1']),
    'HEII1640_BR': (1640.42, ['HeII1640_br_1']),
    'SIIV_OIV':    ((1402.06+1396.76)/2, ['SiIV_OIV1_1', 'SiIV_OIV2_1']),
    'OI1304':      (1304.35, ['OI1304_1']),
    'LYA':         (1215.67, ['Lya_br_1', 'Lya_br_2', 'Lya_br_3']),
    'NV1240':      (1240.42, ['NV1240_1']),
}

# VAC line column -> (local-fit complex name, Gaussian name, rest-frame vacuum wavelength [A]).
# Each of these lines is fitted in its own window with a linear continuum + one Gaussian.
LOCAL_LINES = {
    'NEV3346':   ('NeV3346',   'NeV3346',   3346.8220),
    'NEV3426':   ('NeV',       'NeV3426',   3426.85),
    'FEVII3586': ('FeVII3586', 'FeVII3586', 3587.3436),
    'OII3728':   ('OII',       'OII3728',   3728.48),
    'FEVII3759': ('FeVII3759', 'FeVII3759', 3759.9883),
    'CAII3934':  ('CaII',      'CaII3934',  3934.78),
    'FEXIV5303': ('FeXIV5303', 'FeXIV5303', 5304.3355),
    'FEVII5721': ('FeVII5721', 'FeVII5721', 5722.2871),
    'FEVII6087': ('FeVII6087', 'FeVII6087', 6088.6852),
    'FEX6374':   ('FeX6374',   'FeX6374',   6376.2624),
    'SXII7611':  ('SXII7611',  'SXII7611',  7613.0954),
    'FEXI7892':  ('FeXI7892',  'FeXI7892',  7893.9712),
}

LINE_PROP_NAMES = ['Peak wavelength', 'Flux', 'LogL', 'FWHM', 'EW', 'Centroid']
LINE_PROP_UNITS = ['A', '1e-17 erg/s/cm2', 'erg/s', 'km/s', 'A', 'A']

DESI_DWAVE = 0.8          # A, DESI coadd pixel size (observed frame)
LINE_COVER_DV = 1000.     # km/s, half-width of the window around the line that must be covered


def MC_1sigma_err(ip_arr):
    "Half of the 16-84 percentile spread"
    return (np.percentile(ip_arr, 84)-np.percentile(ip_arr, 16))/2


def check_line_coverage(center, z, wave_good, dv=LINE_COVER_DV, min_frac=0.5):
    """
    True if the +-dv (km/s) window around the rest-frame line center contains more than min_frac of the
    pixels it should have on the DESI grid. Lines failing this are not reported in the VAC.
    """
    wave_lo = center * (1 - dv/C_KMS)
    wave_hi = center * (1 + dv/C_KMS)
    n_expect = (wave_hi - wave_lo) * (1 + z) / DESI_DWAVE
    n_good = np.sum((wave_good >= wave_lo) & (wave_good <= wave_hi))
    return n_good > min_frac * n_expect


def compute_lineprop(DL, center, line_params, conti_func):
    """
    Properties of a line made of one or more Gaussians.
    DL: luminosity distance [cm]; center: rest wavelength of the line [A]
    line_params: flat array [scale1, ln(center1), sigma1, scale2, ...]
    conti_func: continuum model as a function of rest wavelength (from make_continuum)
    Returns (peak [A], flux [1e-17 erg/s/cm2], logL [erg/s], FWHM [km/s], EW [A], centroid [A]).
    """
    line_params = np.asarray(line_params, dtype=float)
    if not np.any(line_params[::3]):
        return 0., 0., 0., 0., 0., 0.
    pp = line_params.reshape(-1, 3)
    cen, sig = pp[:, 1], pp[:, 2]

    left = np.min(cen-5*sig)
    right = np.max(cen+5*sig)
    disp = 1.e-5*np.log(10.)
    npix = int((right-left)/disp)

    xx = np.linspace(left, right, npix)
    xx_linear = np.exp(xx)
    yy_line = Manygauss(xx, pp)
    yy_conti = conti_func(xx_linear)

    # peak wavelength
    ypeak = yy_line.max()
    peak = np.exp(xx[np.argmax(abs(yy_line))])

    # FWHM in km/s: outermost half-maximum crossings, linearly interpolated between grid points
    yy_half = abs(yy_line)-abs(ypeak/2)
    ind_cross = np.where(np.sign(yy_half[:-1]) != np.sign(yy_half[1:]))[0]
    if len(ind_cross) == 0:
        return 0., 0., 0., 0., 0., 0.
    i_l, i_r = ind_cross[0], ind_cross[-1]
    fwhm_left = xx[i_l] - yy_half[i_l]*(xx[i_l+1]-xx[i_l])/(yy_half[i_l+1]-yy_half[i_l])
    fwhm_right = xx[i_r] - yy_half[i_r]*(xx[i_r+1]-xx[i_r])/(yy_half[i_r+1]-yy_half[i_r])
    fwhm = abs(np.exp(fwhm_left)-np.exp(fwhm_right))/center*C_KMS

    area = integrate.trapezoid(yy_line, xx_linear)
    lambda1 = integrate.trapezoid(yy_line*xx_linear, xx_linear)
    ew = integrate.trapezoid(np.abs(yy_line/yy_conti), xx_linear)
    logL = np.log10(4*np.pi*DL**2*abs(area)*1e-17)
    centroid = lambda1/area
    return peak, area, logL, fwhm, ew, centroid


def get_line_prop(linename, op_data, op_data_MC, op_data_spec=None, verbose=True):
    """
    Properties of one line from the global complex fits, as stored in the VAC.
    linename: a key of LINE_LIST (e.g. 'CIV', 'MGII_BR', 'HBETA_BR')
    op_data_spec: if given, apply the same spectral coverage check as the VAC
    Returns (prop, prop_err): arrays of [peak, flux, logL, FWHM, EW, centroid] and their MC uncertainties.
    Lines not fitted / not covered return 0 with errors -1.
    """
    center, sublines = LINE_LIST[linename]
    prop, prop_err = np.zeros(6), -np.ones(6)
    z = op_data['redshift'][0]
    dl = cosmo.luminosity_distance(z).cgs.value
    conti_func = make_continuum(get_conti_para(op_data))

    pp = get_gauss(op_data, sublines, positive_only=False)
    covered = True if op_data_spec is None else \
        check_line_coverage(center, z, get_spectrum(op_data_spec)[0])
    if len(pp) > 0 and covered:
        prop = np.array(compute_lineprop(dl, center, pp.flatten(), conti_func))
        if not np.all(prop == 0):
            pp_mc = np.zeros((len(op_data_MC), 6))
            for nmc in range(len(op_data_MC)):
                pp_tmp = get_gauss(op_data, sublines, op_data_MC, nmc, positive_only=False)
                pp_mc[nmc] = compute_lineprop(dl, center, pp_tmp.flatten(), conti_func)
            ind_fit = pp_mc[:, 0] > 0
            if np.any(ind_fit):
                prop_err = np.array([MC_1sigma_err(pp_mc[ind_fit, k]) for k in range(6)])
    if verbose:
        print_line_prop(linename, prop, prop_err)
    return prop, prop_err


def get_local_line_prop(linename, op_data, op_data_MC=None, verbose=True):
    """
    Properties of one locally fitted line (narrow/coronal lines, CaII K absorption), as stored in the VAC.
    linename: a key of LOCAL_LINES (e.g. 'NEV3426', 'OII3728', 'CAII3934')
    Model: one Gaussian in ln(lambda) + a linear continuum c0 + c1 * v/1000, where v is the velocity
    offset from the line wavelength in km/s (<comp>_PL_norm = c0, <comp>_PL_slope = c1).
    Returns (prop, prop_err, detect); detect = 1 if the line passes the detection criteria.
    Uncertainties come from the bootstrap trials in extension 2 when they were saved, otherwise they
    are propagated from the saved parameter errors.
    """
    comp, gname, lam = LOCAL_LINES[linename]
    prop, prop_err, detect = np.zeros(6), -np.ones(6), 0
    if gname+'_1_scale' not in op_data.names or op_data[gname+'_1_scale'][0] == 0:
        if verbose:
            print_line_prop(linename, prop, prop_err)
        return prop, prop_err, detect
    dl = cosmo.luminosity_distance(op_data['redshift'][0]).cgs.value

    def _props(a, cw, sg, c0, c1):
        peak = np.exp(cw)
        flux = a * sg * np.sqrt(2*np.pi) * np.exp(cw + 0.5 * sg ** 2)
        logl = np.log10(4.0 * np.pi * dl ** 2 * np.maximum(np.abs(flux), 1e-300) * 1e-17)
        fwhm = C_KMS * 2.0 * np.exp(cw) * np.sinh(sg * np.sqrt(2.0 * np.log(2.0))) / lam
        cont = c0 + c1 * C_KMS * (peak / lam - 1.0) / 1000.0
        ew = np.where(cont > 0, flux / np.where(cont > 0, cont, 1.), np.nan)
        cent = np.exp(cw + 1.5 * sg ** 2)
        return np.array([peak, flux, logl, fwhm, ew, cent])

    keys = [gname+'_1_scale', gname+'_1_centerwave', gname+'_1_sigma', comp+'_PL_norm', comp+'_PL_slope']
    par = [op_data[k][0] for k in keys]
    prop = _props(*par)
    detect = int(op_data[gname+'_1_detect'][0])

    if op_data_MC is not None and keys[0]+'_MC' in op_data_MC.names:
        draws = _props(*[np.asarray(op_data_MC[k+'_MC'], dtype=float) for k in keys])
        prop_err = np.array([MC_1sigma_err(d[np.isfinite(d)]) for d in draws])
    else:
        # independent-error propagation by central differences
        err = np.array([op_data[k+'_err'][0] for k in keys])
        var = np.zeros(6)
        for k in range(5):
            if np.isfinite(err[k]) and err[k] > 0:
                up, dn = np.array(par, dtype=float), np.array(par, dtype=float)
                up[k] += err[k]
                dn[k] -= err[k]
                var += np.nan_to_num((_props(*up) - _props(*dn)) / 2.0) ** 2
        prop_err = np.sqrt(var)
    if verbose:
        print_line_prop(linename, prop, prop_err)
    return prop, prop_err, detect


def print_line_prop(linename, prop, prop_err):
    print(linename)
    for name, unit, v, e in zip(LINE_PROP_NAMES, LINE_PROP_UNITS, prop, prop_err):
        print(f'  {name:16s} {v:12.3f} +- {e:9.3f}  [{unit}]')


# ------------------------------------------------------------------------------------------------
# QA plot
# ------------------------------------------------------------------------------------------------
# Gaussian components belonging to each complex, and the plotting window of each complex
COMPLEX_SUBLINES = {
    'Ha': ['Halpha_br_1', 'Halpha_br_2', 'Halpha_br_3', 'Halpha_na_1', 'NII6549_1', 'NII6585_1', 'SII6718_1', 'SII6732_1'],
    'Hb': ['Hbeta_br_1', 'Hbeta_br_2', 'Hbeta_br_3', 'Hbeta_na_1', 'HeII4687_br_1', 'HeII4687_na_1',
           'OIII5007w_1', 'OIII5007c_1', 'OIII4959w_1', 'OIII4959c_1'],
    'HgHb': ['Hgamma_br_1', 'Hgamma_na_1', 'Hdelta_br_1', 'Hdelta_na_1'],
    'MgII': ['MgII_br_1', 'MgII_br_2', 'MgII_na_1'],
    'CIII': ['CIII_br_1', 'CIII_br_2', 'SiIII1892_1', 'AlIII1857_1', 'SiII1816_1', 'NIII1750_1', 'NIV1718_1'],
    'CIV': ['CIV_br_1', 'CIV_br_2', 'CIV_br_3', 'HeII1640_1', 'HeII1640_br_1', 'OIII1663_1', 'OIII1663_br_1'],
    'SiIV': ['SiIV_OIV1_1', 'SiIV_OIV2_1', 'OI1304_1', 'CII1335_1'],
    'Lya': ['Lya_br_1', 'Lya_br_2', 'Lya_br_3', 'NV1240_1'],
}
COMPLEX_RANGE = {
    'Ha': (6300, 6800), 'Hb': (4700, 5100), 'HgHb': (4000, 4500), 'MgII': (2700, 2900),
    'CIII': (1700, 1970), 'CIV': (1500, 1700), 'SiIV': (1280, 1450), 'Lya': (1150, 1290),
}
COMPLEX_LABEL = {'Ha': r'H$\alpha$', 'Hb': r'H$\beta$', 'HgHb': r'H$\gamma$+H$\delta$', 'MgII': 'MgII',
                 'CIII': 'CIII]', 'CIV': 'CIV', 'SiIV': 'SiIV', 'Lya': r'Ly$\alpha$'}

# line IDs marked in the top panel
LINE_ID = [(r'Ly$\alpha$', 1215.67), ('NV', 1240.42), ('OI', 1304.35), ('CII', 1335.30), ('SiIV', 1399.41),
           ('CIV', 1549.06), ('HeII', 1640.42), ('CIII]', 1908.73), ('MgII', 2798.75), ('[NeV]', 3426.84),
           ('[OII]', 3728.48), (r'H$\delta$', 4102.89), (r'H$\gamma$', 4341.68), (r'H$\beta$', 4862.68),
           ('[OIII]', 5008.24), (r'H$\alpha$+[NII]', 6564.61), ('[SII]', 6725.48)]


def get_complexes(op_data):
    "Line complexes fitted globally for this object, ordered blue to red"
    comps = [op_data[n][0].strip() for n in op_data.names
             if n.endswith('_complex_name') and not n.startswith('loc_')]
    comps = [c for c in comps if c in COMPLEX_SUBLINES]
    return sorted(set(comps), key=lambda c: COMPLEX_RANGE[c][0])


def complex_redchi2(op_data, cname):
    "Reduced chi2 of the fit of one complex"
    for n in op_data.names:
        if n.endswith('_complex_name') and not n.startswith('loc_') and op_data[n][0].strip() == cname:
            return op_data[n[:-len('_complex_name')]+'_line_red_chi2'][0]
    return np.nan


def plot_qa(op_data, op_data_spec, title=None, conti_windows=True):
    """
    QA plot of a PyQSOFit result: full spectrum with the model components and residuals (top),
    and a zoom-in on each fitted line complex with the individual Gaussians (bottom).
    For objects with a host decomposition the observed spectrum is shown together with the host model,
    and the continuum/total models include the host; the residuals and zoom-ins use the host-free spectrum.
    """
    wave_full, flux_full, err_full = get_spectrum(op_data_spec)
    wave, flux, err = get_spectrum(op_data_spec, hostfree=True)
    host = has_host(op_data_spec)
    host_model = np.asarray(op_data_spec['flux_pca_host'][:len(wave)], dtype=float) if host else np.zeros_like(wave)

    conti_para = get_conti_para(op_data)
    conti_model = make_continuum(conti_para)(wave)
    pl_poly_model = continuum_PL_poly(wave, conti_para)
    fe_model = conti_model - pl_poly_model
    line_flux = line_model(op_data, wave)

    wave_line_rej = np.asarray(op_data_spec['wave_line_rej_abs'], dtype=float)
    wave_conti_rej = np.asarray(op_data_spec['wave_conti_rej_abs'], dtype=float)
    ind_abs = np.isin(wave, wave_conti_rej[wave_conti_rej != 0]) | np.isin(wave, wave_line_rej[wave_line_rej != 0])

    complexes = get_complexes(op_data)
    ncomp = max(len(complexes), 1)

    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_axes([0.08, 0.625, 0.90, 0.325])
    ax_resid = fig.add_axes([0.08, 0.54, 0.90, 0.075], sharex=ax1)
    if title is None:
        title = (f"{op_data['ra'][0]:.4f}, {op_data['dec'][0]:.4f}; z = {op_data['redshift'][0]:.4f}")
    ax1.set_title(title, fontsize=16)

    if conti_windows and os.path.exists(QSOPAR_FILE):
        cw = fits.getdata(QSOPAR_FILE, 'CONTI_WINDOWS')
        for k, (wlo, whi) in enumerate(zip(cw['min'], cw['max'])):
            if whi > wave_full.min() and wlo < wave_full.max():
                ax1.axvspan(wlo, whi, color='gray', alpha=0.2, lw=0, zorder=0,
                            label='continuum window' if k == 0 else None)
    ax1.errorbar(wave_full, flux_full, yerr=err_full, color='k', ecolor='silver', label='data', zorder=1, lw=0.6)
    if host:
        ax1.plot(wave, host_model, lw=2, c='saddlebrown', zorder=2, label='host')
    ax1.plot(wave, fe_model, lw=2, c='#80724f', zorder=3, label='FeII')
    ax1.plot(wave, pl_poly_model+host_model, lw=2, c='orange', ls='--', zorder=4)
    ax1.plot(wave, conti_model+line_flux+host_model, lw=2.5, c='red', label='PyQSOFit', zorder=5)
    ax1.plot(wave, conti_model+host_model, c='orange', lw=2, label='continuum', zorder=5)
    ax1.plot(wave[ind_abs], flux[ind_abs]+host_model[ind_abs], ls='', marker='x', ms=4, alpha=0.6,
             color='#4c7bd8', label='rejected pixels', zorder=6)

    data_lo, data_hi = np.nanpercentile(medfilt(flux_full, 5), [1, 99.5])
    ax1.set_xlim(wave_full.min(), wave_full.max())
    ax1.set_ylim(min(-0.1*data_hi, data_lo), 1.4*data_hi)
    for name, wv in LINE_ID:
        if wave_full.min() + 10 < wv < wave_full.max() - 10:
            ax1.axvline(wv, color='grey', ls=':', lw=1, zorder=0)
            ax1.text(wv, 1.37*data_hi, name, rotation=90, fontsize=10, color='#676767',
                     ha='center', va='top', zorder=5)
    ax1.axhline(0, c='grey', ls='--', zorder=1)
    ax1.legend(loc='upper right', ncol=3, edgecolor='None', facecolor='w', fontsize=10, framealpha=0.8)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.text(-0.06, -0.15, r'$f_{\lambda}$ [$\rm 10^{-17}\,erg\,s^{-1}\,cm^{-2}\,\AA^{-1}$]', fontsize=16,
             transform=ax1.transAxes, rotation=90, ha='center', rotation_mode='anchor')

    resid = flux - (conti_model + line_flux)
    ax_resid.plot(wave, resid, color='k', lw=0.5, zorder=1)
    ax_resid.axhline(0, c='grey', ls='-', lw=2, zorder=5)
    pad = 4*np.nanstd(resid)
    ax_resid.set_ylim(-pad, pad)
    ax_resid.set_ylabel('resid.', fontsize=12)
    ax_resid.set_xlabel(r'Rest-frame wavelength [$\rm \AA$]')

    fig_dx = 0.045
    fig_x = (0.90-(ncomp-1)*fig_dx)/ncomp
    for n, cname in enumerate(complexes):
        rng = COMPLEX_RANGE[cname]
        ind_line = (wave > rng[0]) & (wave < rng[1])
        lw_line = np.log(wave[ind_line])
        axn = fig.add_axes([0.08+n*fig_x+n*fig_dx, 0.08, fig_x, 0.36])
        ydata = flux[ind_line]-conti_model[ind_line]
        axn.errorbar(wave[ind_line], ydata, yerr=err[ind_line], color='k', ecolor='silver', zorder=1, lw=0.7)
        axn.plot(wave[ind_line], line_flux[ind_line], color='red', lw=2, zorder=4)
        for subline in COMPLEX_SUBLINES[cname]:
            pp = get_gauss(op_data, [subline])
            if len(pp) > 0:
                # narrow components (FWHM < 1200 km/s) in blue, broad components in green
                fwhm = 2*np.sqrt(2*np.log(2))*(np.exp(pp[0, 2])-1)*C_KMS
                axn.plot(wave[ind_line], Manygauss(lw_line, pp), color='b' if fwhm < 1200 else 'seagreen',
                         lw=1.5, zorder=2 if fwhm < 1200 else 3)
        axn.text(0.95, 0.96, r'$\chi^2_r=$'+f'{complex_redchi2(op_data, cname):.2f}', fontsize=12,
                 va='top', ha='right', transform=axn.transAxes)
        axn.text(0.05, 0.96, COMPLEX_LABEL[cname], fontsize=14, transform=axn.transAxes, va='top', ha='left')
        axn.axhline(0, c='grey', ls='--', zorder=1)
        axn.set_xlim(rng)
        if np.sum(ind_line) > 5:
            lo, hi = np.nanpercentile(ydata, [2, 98])
            pad = 0.15*(hi-lo+1e-3)
            axn.set_ylim(min(lo-pad, -0.2*(hi+pad)), hi+2*pad)
        axn.minorticks_on()
        if n == 0:
            axn.set_ylabel('line only', fontsize=12)
        if n == ncomp//2:
            axn.set_xlabel(r'Rest-frame wavelength [$\rm \AA$]')
    return fig


# ------------------------------------------------------------------------------------------------
# catalog-level plotting helpers
# ------------------------------------------------------------------------------------------------
def get_levels(count_arr, quantile):
    "Density levels that enclose 1-quantile of the total counts"
    values = np.ravel(count_arr)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - np.asarray(quantile))
    return np.take(sorted_values, idx, mode="clip")


def plot_density(ax, x, y, bins=50, range=None, percent=(0.003, 0.05, 0.32), smooth=0.25,
                 cmap='Greys', clabel=True):
    """
    Scatter points in the outskirts, plus a 2D histogram and contours enclosing
    (1-percent) of the objects (99.7%, 95% and 68% by default).
    """
    counts, xbins, ybins = np.histogram2d(x, y, bins=bins, range=range)
    xm, ym = np.meshgrid((xbins[:-1]+xbins[1:])/2, (ybins[:-1]+ybins[1:])/2)
    counts = counts.T
    levels = np.unique(get_levels(counts, np.array(percent)))

    ax.plot(x, y, ls='None', marker='.', mec='None', mfc='k', ms=2, alpha=0.1, zorder=1, rasterized=True)
    masked = np.where(counts <= levels.min(), np.nan, counts)
    ax.pcolormesh(xm, ym, np.log10(masked), vmin=np.log10(levels.min()), vmax=np.log10(counts.max()),
                  cmap=cmap, zorder=2, shading='auto', rasterized=True)
    ct = ax.contour(xm, ym, gaussian_filter(counts, smooth), levels=levels, linewidths=2,
                    colors='silver', zorder=4)
    if clabel:
        # the smallest percent (largest enclosed fraction) gives the lowest level
        labels = {lv: f'{(1-p)*100:g}%' for lv, p in zip(levels, sorted(percent))}
        ax.clabel(ct, ct.levels, fmt=labels, fontsize=8, inline=True)
    return ax
