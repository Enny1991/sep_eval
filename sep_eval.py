import mir_eval
import numpy as np
import librosa
import soundfile as sf
from pystoi.stoi import stoi as source_stoi
from sdr import calc_sdr
import subprocess
from copy import deepcopy as dc


def full_eval(deg, ref, fs=8000, avg=True):
    return {'pesq': np.mean(pesq(deg, ref, fs)) if avg else pesq(deg, ref, fs), \
           'stoi': np.mean(stoi(deg, ref, fs)) if avg else stoi(deg, ref, fs), \
           'e_stoi': np.mean(stoi(deg, ref, fs, extended=True)) if avg else stoi(deg, ref, fs, extended=True), \
           'sdr': np.mean(sdr(deg, ref, fs)) if avg else sdr(deg, ref, fs)}

    
def pesq(__deg, __ref, fs=8000):
    deg = dc(__deg)
    ref = dc(__ref)
    if fs != 8000 and fs != 16000:
        raise ValueError('Supported sampling frequency is only 8000 or 16000')
    if len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    
    rets = []
    for _deg, _ref in zip(deg, ref):
        _deg /= np.max(np.abs(_deg))
        sf.write('/tmp/deg.wav', _deg, fs)
        sf.write('/tmp/ref.wav', _ref, fs)
        p = subprocess.Popen("pesq /tmp/ref.wav /tmp/deg.wav +{}".format(fs), stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        ## Wait for date to terminate. Get return returncode ##
        p_status = p.wait()
        try:
            rets.append(float(output.split(' ')[-1]))
        except:
            rets.append(np.nan)
    return rets

def stoi(__deg, __ref, fs=8000, extended=False):
    deg = dc(__deg)
    ref = dc(__ref)
    if len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    return [source_stoi(_deg, _ref, fs, extended=extended) for _deg, _ref in zip(deg, ref)]


def sdr(__deg, __ref, fs=8000):
    deg = dc(__deg)
    ref = dc(__ref)
    if len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    s = calc_sdr(deg, ref)
    return s.squeeze()

def bss_eval(degs, refs):
    if len(degs.shape) == 2:
        degs = np.expand_dims(degs, 0)
        refs = np.expand_dims(refs, 0)
    return [mir_eval.separation.bss_eval_sources(_degs, _refs) for _degs, _refs in zip(degs, refs)]
