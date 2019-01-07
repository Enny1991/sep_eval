from __future__ import print_function
import mir_eval
import numpy as np
import soundfile as sf
from pystoi.stoi import stoi as source_stoi
import subprocess
import os
from copy import deepcopy as dc
import sys


def calc_sdr(degraded, reference):
    """
    batch-wise SDR calculation for one audio file.
    estimation: (batch, n_sample)
    origin: (batch, n_sample)
    """
    if len(degraded.shape) == 1:
        degraded = np.expand_dims(degraded, 0)
        reference = np.expand_dims(reference, 0)
    origin_power = np.sum(reference ** 2, 1, keepdims=True)  # (batch, 1)

    scale = np.sum(reference * degraded, 1, keepdims=True) / origin_power  # (batch, 1)

    est_true = scale * reference  # (batch, n_sample)
    est_res = degraded - est_true  # (batch, n_sample)

    true_power = np.sum(est_true ** 2, 1)
    res_power = np.sum(est_res ** 2, 1)

    return 10 * np.log10(true_power) - 10 * np.log10(res_power)  # (batch, 1)


def full_eval(degraded, reference, fs=8000, average=False, verbose=False):
    return {'pesq': pesq(degraded, reference, fs, verbose=verbose, average=average),
            'stoi': stoi(degraded, reference, fs, average=average),
            'e_stoi': stoi(degraded, reference, fs, extended=True, average=average),
            'sdr': sdr(degraded, reference, average=average)}


def pesq(degraded, reference, fs=8000, verbose=False, average=False):
    """
    Help with installing pesq:
    Download the binaries from here: https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200511-I!Amd2!SOFT-ZST-E&type=items
    unzip and cd to Software/P862_annex_A_2005_CD/source
    the run
    gcc -o PESQ *.c -lm
    to compile
    add this location to path
    DONE!
    """
    deg = dc(degraded)
    ref = dc(reference)
    if fs != 8000 and fs != 16000:
        raise ValueError('Supported sampling frequency is only 8000 or 16000')
    if not type(degraded) is list and len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)

    _pesq = []
    for _deg, _ref in zip(deg, ref):
        _deg /= np.max(np.abs(_deg))
        sf.write('/tmp/deg.wav', _deg, fs)
        sf.write('/tmp/ref.wav', _ref, fs)
        p = subprocess.Popen("cd /tmp/; pesq ref.wav deg.wav +{}".format(fs), stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        if verbose:
            print("Output: {}".format(output))
            print("Error: {}".format(err))
            print("Status: {}".format(p_status))
        try:
            _pesq.append(float(output.split(' ')[-1][:6]))
        except ValueError:
            _pesq.append(np.nan)
    if os.path.exists('./_pesq_itu_results.txt'):
        os.remove('_pesq_itu_results.txt')
    if os.path.exists('./_pesq_results.txt'):
        os.remove('_pesq_results.txt')
    all_res = np.array(_pesq).squeeze()
    return np.mean(all_res) if average else all_res


def stoi(degraded, reference, fs=8000, extended=False, average=False):
    deg = dc(degraded)
    ref = dc(reference)
    if not type(degraded) is list and len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    all_res = np.array([source_stoi(_deg, _ref, fs, extended=extended) for _deg, _ref in zip(deg, ref)]).squeeze()
    return np.mean(all_res) if average else all_res


def sdr(degraded, reference, average=False):
    deg = dc(degraded)
    ref = dc(reference)
    if not type(degraded) is list and len(deg.shape) == 1:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    all_res = np.array([calc_sdr(_deg, _ref) for _deg, _ref in zip(deg, ref)]).squeeze()
    return np.mean(all_res) if average else all_res


def bss_eval(degraded, reference, average=False):
    deg = dc(degraded)
    ref = dc(reference)
    if not type(degraded) is list and len(deg.shape) == 2:
        deg = np.expand_dims(deg, 0)
        ref = np.expand_dims(ref, 0)
    all_res = {'sdr': [], 'sir': [], 'sar': [], 'perm': []}
    for _deg, _ref in zip(deg, ref):
        _res = mir_eval.separation.bss_eval_sources(_deg, _ref)
        all_res['sdr'].append(_res[0])
        all_res['sir'].append(_res[1])
        all_res['sar'].append(_res[2])
        all_res['perm'].append(_res[3])

    all_res['sdr'] = np.array(all_res['sdr']).squeeze()
    all_res['sir'] = np.array(all_res['sir']).squeeze()
    all_res['sar'] = np.array(all_res['sar']).squeeze()
    all_res['perm'] = np.array(all_res['perm']).squeeze()
    if average:
        all_res['sdr'] = np.mean(all_res['sdr'], 0)
        all_res['sir'] = np.mean(all_res['sir'], 0)
        all_res['sar'] = np.mean(all_res['sar'], 0)
    return all_res
