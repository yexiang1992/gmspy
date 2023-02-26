import matplotlib.pyplot as plt
import gmspy as gm
import numpy as np

import pywt

class PulseData(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<PulseData ' + dict.__repr__(self) + '>'


# This function returns the continuous wavelet transform coeficients for
# the input signat at the specified time and scales
def cont_wavelet_trans(signal, dt, scales, wname):
    cwt_coefs = pywt.cwt(signal, scales, wname)
    return cwt_coefs


def analyze_record(signal,dt,col,row,scales,wname):
    # Function to classify an extracted pulse as pulse-like or non pulse-like
    # 
    # Input 
    #
    # signal : the velocity time history of original ground motion
    # col, row : location of the identified pulse
    # scales: scales at which wavelet transform was computed
    # wname : name of the wavelet used for wavlet analysis
    ###############
    #
    # Output
    #
    # pulse_data : A structure with details about the classification results.

    num_coefs = 10 # number of wavelet coefs used to construct the pulse
    # refine the scales to get a better resolution of pulse (zoom in)
    num_scales = len(scales)
    scales = [scales(max(1, row-1)): scales(min(num_scales, row+1))]
    cwt_coefs = pywt.cwt(signal,scales, wname)
    z=abs(cwt_coefs(:,col)) # only use the coefficients at the same time as the original pulse
    row = find(z==max(z))
    pulse_scale = scales(row)

    time = dt:dt:length(signal)*dt
    wtype = wavemngr('type',wname)
    iter = 4

    switch wtype
        case 1 , [nul,psi,xval] = wavefun(wname,iter)
        case 2 , [nul,psi,nul,nul,xval] = wavefun(wname,iter)
        case 3 , [nul,psi,xval] = wavefun(wname,iter)
        case 4 , [psi,xval] = wavefun(wname,iter)
        case 5 , [psi,xval] = wavefun(wname,iter)
    end


    resid_th = signal
    pulse_th = zeros(size(signal))

    for i = 1:num_coefs
        [coefs(1,i), ~, col(1,i), Tp] = fn_extract_one_wavelet(resid_th, dt, pulse_scale, col(1))
        
        # compute the residual
        basis = xval.*(pulse_scale)
        basis = basis + (col(1,i) - median(basis))
        basis = basis.*dt
        y_vals = psi * coefs(1,i) / sqrt(pulse_scale)
        delta = basis(2) - basis(1)
        num_pads = ceil((max(time) - max(basis))/delta)
        final_basis = [0:delta:min(basis-0.00001) basis (max(basis)+delta:delta:max(basis)+delta*num_pads)]
        final_yvals = [zeros(size(0:delta:min(basis-0.00001))) y_vals zeros(1,num_pads)]
        pulse_th = pulse_th + interp1(final_basis,final_yvals,time)
        pulse_th(find(isnan(pulse_th))) = 0
        resid_th = signal - pulse_th
    end

    # Check if the pulse is late

    signal_E = cumsum(signal.^2)/(sum(signal.^2))*100
    pulse_E = cumsum(pulse_th.^2)/(sum(pulse_th.^2))*100
    index = find(pulse_E <= 5)
    lateTime = signal_E(index(end))

    late = lateTime >= 17

    # sum of the coefficients of the DWT of the original signal and the final
    # residual. This sum is also equal to the energy of the ground motion.
    pulse_data = PulseData(dict())
    pulse_data.dwt_orig = sum(sum(abs(dwt(signal,wname))))
    pulse_data.dwt_resid = sum(sum(abs(dwt(resid_th,wname))))
    pulse_data.dwt_squared_orig = sum(sum( dwt(signal,wname).^2  ))
    pulse_data.dwt_squared_resid = sum(sum( dwt(resid_th,wname).^2  ))

    # compute the "Pulse Indicator" based on the PGV ratio and energy ratio
    PGV = max(abs(signal))
    PGV_resid = max(abs(resid_th))
    pgvRatio = PGV_resid/PGV
    energyRatio = pulse_data.dwt_squared_resid/pulse_data.dwt_squared_orig
    pc = 0.63*pgvRatio + 0.777*energyRatio # the principle component
    # center the variables
    P = (pc-1.208421)/0.2462717
    V = (PGV-11.58861)/18.88015
    pulse_indicator = -7.817 - 0.5679*P^2 - 0.1516*V^2 - 3.0253*P -1.7396*V -2.7156*P*V

    # identify whether the record contains a significant pulse
    is_pulse = (pulse_indicator>0) & ~late


    # build a structure to hold the decomposition data
    pulse_data.dt = dt # time step of the ground motion recording
    pulse_data.num_pts = length(signal) # number of points in the recording
    pulse_data.Tp = Tp # pulse period
    pulse_data.wavelet_name = wname # name of the wavelet used for analysis
    pulse_data.pulse_scale = pulse_scale # scale at which the largest wavelet was found
    pulse_data.rows = col # rows where wavelets were extracted
    pulse_data.coefs = coefs # coefficients for the extracted wavelets
    pulse_data.PGV = PGV # Peak ground velocity of the original ground motion
    pulse_data.PGV_resid = PGV_resid # Peak ground velocity of the residual ground motion
    pulse_data.delta_energy_t= delta # if this is negative, then the pulse is late-arriving
    pulse_data.late = late # is the pulse late arriving
    pulse_data.pulse_indicator = pulse_indicator # the score from the logistic regression used for classification
    pulse_data.is_pulse = is_pulse # classification of the record
    pulse_data.signal = signal
    pulse_data.pulse_th = pulse_th
    pulse_data.resid_th = resid_th

# function to extract one wavelet

function [coef, pulse_scale, col, Tp] = fn_extract_one_wavelet(signal, dt, pulse_scale, pulse_row)
    # defined parameters
    Tp_min = 0.25 # minimum period to consider for pulses
    Tp_max = 15 # maximum period to consider for pulses
    row_range = 10/25 # range of row locations in which to look for supplementary wavelets
    wname = 'db4' # use Daubechies Wavelet of fourth order for the transform

    # set the scales at which to perform the CWT
    # scale and range of rows to consider are already known
    scales = pulse_scale
    row_indices = [max(1, pulse_row - ceil(scales*row_range)) : min(length(signal), pulse_row + ceil(scales*row_range))]
    # peform the Continuous Wavelet Transform
    cwt_coefs = cwt(signal,scales, wname)

    # find max coefficient

    z=max(max(abs(cwt_coefs(1,row_indices))))
    row = 1
    col = find(z==(abs(cwt_coefs(1,:))))

    # the value of the coeficient associated with the extracted pulse
    coef = cwt_coefs(row, col)

    # pulse period (Tp)
    Tp = 1./scal2frq(pulse_scale,wname,dt)



ts, acc = gm.load_gm_examples("Kobe")
dt = ts[1] - ts[0]
