# Raman spectrum processing - version 4
# 0. Plot the raw data
# 1. Remove spikes from cosmic rays using z-scores
# 2. Baseline estimation and subtraction usingasymmetric least squares smoothing
# 3. Savitsky Golay smoothing of data for presentation - unnecessary additional step
# 4. Gaussian filtering to clean up curves for identification of the peaks
# 5. Estimation of peak parameters (height, centre, width) for Lorentzian curve fitting - currently input manually
# 6. Fit Lorentzian curves to the data and find the residuals
# 7. Plot the Lorentzian curves over the data and the residuals
# 8. Save the data

# Loading the required packages:
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec

import scipy as scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy import signal

import pandas as pd
import itertools

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load the raman data
df = pd.read_csv('.txt', delimiter = '\t')


sample = ""
path = ""
output_path = str(path) + str(sample) + "/" + str(sample)

# Preventative step to avoid overwriting pre-existing data
if not os.path.exists(str(path)+str(sample)):
    # Create the directory
    os.makedirs(str(path)+str(sample))
    print("Directory created")
else:
    input_checker = input('\nDirectory already exists. Check before continuing. \nIn the terminal: type y to continue or any other key to exit: >')
    if input_checker == 'y':
        print("\nReduction will run\n")
    else:
        print("\nCheck sample and path\n ")
        os._exit(0)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0. RAW DATA


# Transform the data to a numpy array
wavelength = np.array(df.Wavelength)
intensity = np.array(df.Intensity)

# Plot the raw data spectrum:
fig0 = plt.figure()
plt.plot(wavelength, intensity)
plt.title('Raw data', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Wavelength (cm^-1)', fontsize = 20)
plt.ylabel('Intensity (a.u.)', fontsize = 20)
plt.show()
fig0.tight_layout()
fig0.savefig(os.path.join(str(output_path)+'_raw_spectra.png'), format="png",dpi=1000)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. REMOVE SPIKES FROM COSMIC RAYS
# https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22

# Calculate the modified z-scores of a differentiated spectrum
def modified_z_score(ys):
    ysb = np.diff(ys) # Differentiated intensity values
    median_y = np.median(ysb) # Median of the intensity values
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ysb]) # median_absolute_deviation of the differentiated intensity values
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ysb] # median_absolute_deviationmodified z scores
    return modified_z_scores
    
# Calculate the average values around the point to be replaced.
def spike_removal(y,ma):
    threshold = 7 # binarization threshold
    spikes = abs(np.array(modified_z_score(y))) > threshold
    y_out = y.copy()
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            w = np.arange(i-ma,i+1+ma)
            we = w[spikes[w] == 0]
            y_out[i] = np.mean(y[we])
    return y_out


# The despiking algorithm cannot handle smooth, generated data without noise - despiking should be skipped in these cases
#to_despike = input('\nDespiking needed? \nType y to continue or any other key to continue without despiking >')
#if to_despike == 'y':
intensity_1_despiked = spike_removal(intensity,ma=10)
fig1 = plt.figure()
plt.plot(wavelength, intensity, color = 'black', label = 'Raw spectrum')
plt.plot(wavelength, intensity_1_despiked, color = 'red', label = 'Despiked spectrum')
plt.title('Cosmic ray despiked', fontsize = 15)
plt.xlabel('Wavelength', fontsize = 15)
plt.ylabel('Intensity',  fontsize = 15)
plt.legend()
plt.show()
fig1.tight_layout()
fig1.savefig(os.path.join(str(output_path)+'_despiked_spectra.png'), format="png",dpi=300)
df['1_intensity_despiked'] = intensity_1_despiked.tolist()
#else:
#    intensity_1_despiked = intensity



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. BASELINE ESTIMATION AND SUBTRACTION
# https://towardsdatascience.com/data-science-for-raman-spectroscopy-a-practical-example-e81c56cf25f
# Coding for removal with asymmetric least squares, current parameters force linear baseline
# According to paper: "Baseline Correction with Asymmetric Least Squares Smoothing" 
# by Paul H. C. Eilers and Hans F.M. Boelens. October 21, 2005

# Baseline estimation function:
def baseline_als(y, lam, p, niter=100):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# Parameters
# l = smoothness
# 10² < l < 10⁹  (10^12 forces to linear baseline)
l_smoothness = 1e10

#p = asymmetry, i.e. the weight of the baseline compared to the peaks
#0.001 < p < 0.1
p_asymmetry = 0.01




# Estimation of the baseline:
estimated_baseline = baseline_als(intensity_1_despiked, l_smoothness, p_asymmetry)

# Baseline subtraction:
intensity_2_baseline_subtracted = intensity_1_despiked - estimated_baseline

# Compare the raw data + baseline to data with baseline removed
fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(16,4))

# Plot the raw spectrum with the estimated baseline
ax1.plot(wavelength, intensity_1_despiked, color = 'black', label = 'Raw spectrum' )
ax1.plot(wavelength, estimated_baseline, color = 'red', label = 'Estimated baseline')
ax1.set_title('Baseline estimation', fontsize = 15)
ax1.set_xlabel('Wavelength', fontsize = 15)
ax1.set_ylabel('Intensity',  fontsize = 15)
ax1.legend()

# Plot the spectrum with baseline removed
ax2.plot(wavelength, intensity_2_baseline_subtracted, color = 'black', label = 'Spectrum with baseline subtracted' )
ax2.set_title('Baseline subtracted', fontsize = 15)
ax2.set_xlabel('Wavelength', fontsize = 15)
ax2.set_ylabel('Intensity',  fontsize = 15)
plt.show()




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. SMOOTHING THE SPECTRUM
# Some more information on the implementation of this method can be found here:
# https://nirpyresearch.com/savitzky-golay-smoothing-method/

# Parameters:
window = 9 # window (number of points)
polynomial_order = 2 # polynomial order

intensity_3_savgol_smoothed = savgol_filter(intensity_2_baseline_subtracted, window, polyorder = polynomial_order, deriv=0)

# Plot spectrum after baseline subtraction
fig3 = plt.figure()
plt.plot(wavelength, intensity_2_baseline_subtracted, color = 'black', label = 'Spectrum with baseline removed' )
plt.plot(wavelength, intensity_3_savgol_smoothed, color = 'red', label = 'Smoothed spectrum' )
plt.title('Smoothing raman spectrum', fontsize = 15)
plt.xlabel('Wavelength', fontsize = 15)
plt.ylabel('Intensity',  fontsize = 15)
plt.show()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. GAUSSIAN FILTERING

# Gaussian filtering
intensity_4_gaussian_smoothed = gaussian_filter1d(intensity_2_baseline_subtracted, sigma=11)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. FIND ESTIMATE PEAK PARAMETERS IN GAUSSIAN SMOOTHED DATA

# Find peaks in Gaussian smoothed data
fig5 = plt.figure()
prominence_for_id = 1
sig_peaks_fp, _ = signal.find_peaks(intensity_4_gaussian_smoothed, prominence = prominence_for_id)
plt.plot(wavelength[sig_peaks_fp], intensity_4_gaussian_smoothed[sig_peaks_fp], "ob")
plt.plot(wavelength, intensity_4_gaussian_smoothed)
plt.show()


# Break point to stop the rest of the code running in case of unidentified peaks
correct_peaks = input('\nAre the peaks correctly identified? Type y to continue or any other key to exit: >')
if correct_peaks == 'y':
   print("\nPeaks correctly identified, continuing with fitting\n")
else:
    print("\nChange prominence in find peaks, else manually input parameters\n ")
    os._exit(0)

# Find peak parameters for Lorentzian fitting
peak_centres = np.array(wavelength[sig_peaks_fp])
peak_amps = np.array(intensity_4_gaussian_smoothed[sig_peaks_fp])
half_peak_res = signal.peak_widths(intensity_4_gaussian_smoothed, sig_peaks_fp, rel_height = 0.5)
peak_wids = np.array(half_peak_res[0])

# If peaks cannot correctly identified and need to be done manually, input estimates here to overwrite:
peak_centres = np.array([1350,1600])
peak_amps = np.array([1000,1000])
peak_wids = np.array([100,100])

all_peak_params = np.array(list(itertools.chain(*zip(peak_amps, peak_centres, peak_wids))))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. FIT LORENTZIAN CURVES
# http://emilygraceripka.com/blog/16
# http://astrophysicsformulas.com/astronomy-formulas-astrophysics-formulas/lorentzian-fwhm-calculation/
# https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_how_to_lorentzian.htm
# https://cars.uchicago.edu/xraylarch/fitting/lineshapes.html
# https://stackoverflow.com/questions/76731539/full-width-at-half-parameter-of-lorentzian-fit-with-scipy-curve-fit-in-python
# Best guess curve parameters

# peak_params = amp0, cen0, wid0, amp1, cen1, wid1...
def _xLorentzian(x, *peak_params): 
    sum_lorentz = 0
    for i in range(0, len(peak_params), 3):
        sum_lorentz = sum_lorentz + (peak_params[i]*peak_params[i+2]**2/((x-peak_params[i+1])**2+peak_params[i+2]**2))
    return sum_lorentz

def Lorentz_FWHM(sigma):
    return 2*sigma

# Use Lorentzian functions with optimal values for parameters (popt) and estimated covariance (pcov)
popt_xlorentz, pcov_xlorentz = scipy.optimize.curve_fit(_xLorentzian, wavelength, intensity_3_savgol_smoothed, p0 =[all_peak_params])

# One sigma errors on parameters
perr_xlorentz = np.sqrt(np.diag(pcov_xlorentz))

# Put pars into a usable form
pars_variables = {}
for i in range(0, len(peak_centres)):
    pars_variables['pars_'+str(i)] = popt_xlorentz[i*3:i*3+3]
locals().update(pars_variables)

# Input pars into Lorentzian function to calculate curves
lorentz_variables = {}
count = 0
for key in pars_variables:
    pars_for_calc = pars_variables[key]
    lorentz_variables['lorentz_'+str(count)] = _xLorentzian(wavelength, *pars_for_calc)
    count += 1
locals().update(lorentz_variables)

pars_list = list(pars_variables.values())
concat_pars = np.concatenate(pars_list)

# Calculate area under lorentz curves
count = 0
curve_area = {}
for key in lorentz_variables:
    lorentz_for_area = lorentz_variables[key]
    curve_area['peak_'+str(count)+'_area'] = np.trapz(lorentz_for_area)
    count += 1
curve_list = list(curve_area.values())


peaks = {}
for i in range(len(peak_centres)):
    peak_properties = {}
    for j in range(1):
        peak_properties[f"peak"] = i
        peak_properties[f"amplitude"] = concat_pars[i*3]
        peak_properties[f"amplitude_err"] = perr_xlorentz[i*3]
        peak_properties[f'centre'] = concat_pars[i*3+1]
        peak_properties[f'centre_err'] = perr_xlorentz[i*3+1]
        peak_properties[f'FWHM'] = Lorentz_FWHM(concat_pars[i*3+2])
        peak_properties[f'FWHM_err'] = perr_xlorentz[i*3+2] * 2
        peak_properties[f'HWHM'] = concat_pars[i*3+2]
        peak_properties[f'HWHM_err'] = perr_xlorentz[i*3+2]
        peak_properties[f'area'] = (curve_list[i])
    peaks[f"peak_no_{i}"] = peak_properties


# Calculate residuals
residual_xlorentz = intensity_2_baseline_subtracted - (_xLorentzian(wavelength, *popt_xlorentz))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 7. PLOT CURVES AND RESIDUALS

# Plot Lorentzian fits and residuals
fig7 = plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
ax1 = fig7.add_subplot(gs[0])
ax2 = fig7.add_subplot(gs[1])
gs.update(hspace=0) 

ax1.plot(wavelength, intensity_3_savgol_smoothed, "ro")
ax1.plot(wavelength, _xLorentzian(wavelength, *popt_xlorentz), 'k--')#,\
         #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

# Plot curves
for key in lorentz_variables:
    ax1.plot(wavelength, lorentz_variables[key])
    ax1.fill_between(wavelength, lorentz_variables[key].min(), lorentz_variables[key],  alpha=0.5)

# Plot residuals
ax2.plot(wavelength, residual_xlorentz, "bo")

# Format figure    
ax2.set_xlabel("Wavelength", fontsize=12)
ax1.set_ylabel("Intensity", fontsize=12)
ax2.set_ylabel("Res.", fontsize=12)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))

ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.xaxis.set_major_formatter(plt.NullFormatter())

ax1.tick_params(axis='x',which='major', direction="in", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='x',which='minor', direction="in", top="on", right="on", bottom="off", length=5, labelsize=8)
ax1.tick_params(axis='y',which='major', direction="in", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='y',which='minor', direction="in", top="on", right="on", bottom="on", length=5, labelsize=8)

ax2.tick_params(axis='x',which='major', direction="in", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='x',which='minor', direction="in", top="off", right="on", bottom="on", length=5, labelsize=8)
ax2.tick_params(axis='y',which='major', direction="in", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='y',which='minor', direction="in", top="off", right="on", bottom="on", length=5, labelsize=8)

plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# 8. SAVE DATA


fig2.tight_layout()
fig2.savefig(os.path.join(str(output_path)+'_baselined_spectra.png'), format="png",dpi=300)

fig3.tight_layout()
fig3.savefig(os.path.join(str(output_path)+'_savgol_smoothed_spectra.png'), format="png",dpi=300)

fig5.tight_layout()
fig5.savefig(os.path.join(str(output_path)+'_gaussian_smoothed_peaks_identified.png'), format="png",dpi=300)

fig7.tight_layout()
fig7.savefig(os.path.join(str(output_path)+'_lorentzian_peaks_fitted_residuals.png'), format="png",dpi=300)


with open(str(output_path)+'code_params.txt', 'w') as f:
    print(f"{sample} code parameters\n", end='', file=f)
    print(f"Baseline estimation\n", end='', file=f)
    print(f"Smoothness = {l_smoothness}\n", end='', file=f)
    print(f"Asymmetry = {p_asymmetry}\n", end='', file=f)
    #if to_despike == 'y':
    #    print(f"Despiked\n", end='', file=f)
    #else:
    #    print(f"No despiking\n\n", end='', file=f)
    print("Peak identification\n", end='', file=f)
    print(f"Peak prominence = {prominence_for_id}", end='', file=f)



df['estimated_baseline'] = estimated_baseline.tolist()
df['2_intensity_baseline_subtracted'] = intensity_2_baseline_subtracted.tolist()
df['3_intensity_savgol_smoothed'] = intensity_3_savgol_smoothed.tolist()
df['4_intensity_gaussian_smoothed'] = intensity_4_gaussian_smoothed.tolist()
df['residual_xlorentz'] = residual_xlorentz.tolist()


peak_df = pd.DataFrame.from_dict(peaks, orient='index') # convert dict to dataframe
peak_df.to_csv(r''+str(output_path)+'_peak_parameters.csv', index=False)

df2 = pd.DataFrame.from_dict(lorentz_variables, orient='index')
df2 = df2.transpose()
df_all = pd.concat([df,df2], axis=1)
df_all.to_csv(r''+str(output_path)+'_data_fitting.csv', index=False)