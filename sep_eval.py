from __future__ import print_function
import mir_eval
import numpy as np
import soundfile as sf
from pystoi.stoi import stoi as source_stoi
import subprocess
import os
from copy import deepcopy as dc


def calc_sdr(deg, ref):
    """
    batch-wise SDR calculation for one audio file.
    estimation: (batch, n_sample)
    origin: (batch, n_sample)
    """

    origin_power = np.sum(ref ** 2, 1, keepdims=True)  # (batch, 1)

    scale = np.sum(ref * deg, 1, keepdims=True) / origin_power  # (batch, 1)

    est_true = scale * ref  # (batch, n_sample)
    est_res = deg - est_true  # (batch, n_sample)

    true_power = np.sum(est_true ** 2, 1)
    res_power = np.sum(est_res ** 2, 1)

    return 10 * np.log10(true_power) - 10 * np.log10(res_power)  # (batch, 1)


def full_eval(deg, ref, fs=8000, avg=True):
    return {'pesq': np.mean(pesq(deg, ref, fs)) if avg else pesq(deg, ref, fs),
            'stoi': np.mean(stoi(deg, ref, fs)) if avg else stoi(deg, ref, fs),
            'e_stoi': np.mean(stoi(deg, ref, fs, extended=True)) if avg else stoi(deg, ref, fs, extended=True),
            'sdr': np.mean(sdr(deg, ref)) if avg else sdr(deg, ref)}


def pesq(__deg, __ref, fs=8000, verbose=False):
    """
    Help with installing pesq:
    Download the binaries from here: https://www.itu.int/rec/T-REC-P.862-200511-I!Amd2/en
    unzip and cd to Software/P862_annex_A_2005_CD/source
    the run
    gcc -o PESQ *.c -lm
    to compile
    add this location to path
    DONE!
    """
    deg = dc(__deg)
    ref = dc(__ref)
    if fs != 8000 and fs != 16000:
        raise ValueError('Supported sampling frequency is only 8000 or 16000')
    if len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)

    _pesq = []
    for _deg, _ref in zip(deg, ref):
        _deg /= np.max(np.abs(_deg))
        sf.write('/tmp/deg.wav', _deg, fs)
        sf.write('/tmp/ref.wav', _ref, fs)
        p = subprocess.Popen("pesq /tmp/ref.wav /tmp/deg.wav +{}".format(fs), stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        if verbose:
            print("Output: {}".format(output))
            print("Error: {}".format(err))
            print("Status: {}".format(p_status))
        try:
            _pesq.append(float(output.split(' ')[-1]))
        except ValueError:
            _pesq.append(np.nan)
    os.remove('_pesq_itu_results.txt')
    os.remove('_pesq_results.txt')
    return np.array(_pesq).squeeze()


def stoi(__deg, __ref, fs=8000, extended=False):
    deg = dc(__deg)
    ref = dc(__ref)
    if len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    return np.array([source_stoi(_deg, _ref, fs, extended=extended) for _deg, _ref in zip(deg, ref)]).squeeze()


def sdr(__deg, __ref):
    deg = dc(__deg)
    ref = dc(__ref)
    if len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    s = calc_sdr(deg, ref)
    return s.squeeze()


def bss_eval(__deg, __ref):
    deg = dc(__deg)
    ref = dc(__ref)
    if len(deg.shape) == 2:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    return [mir_eval.separation.bss_eval_sources(_deg, _ref) for _deg, _ref in zip(deg, ref)]
