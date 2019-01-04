from __future__ import print_function
from sep_eval import sep_eval
import soundfile as sf
import argparse
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():

    deg, fs = sf.read('../wavs/degraded.wav')
    ref1, fs = sf.read('../wavs/reference1.wav')
    enh1, fs = sf.read('../wavs/enhanced1.wav')

    ref2, fs = sf.read('../wavs/reference2.wav')
    enh2, fs = sf.read('../wavs/enhanced2.wav')

    if args.measure == 'full':

        print("Full test")

        print("\tBefore separation:")

        eval_deg = sep_eval.full_eval(deg, ref1)

        for measure, score in eval_deg.items():
            print("\t\t{} = {}".format(measure, score))

        print("\tAfter separation")

        eval_enh1 = sep_eval.full_eval(enh1, ref1)

        for measure, score in eval_enh1.items():
            print("\t\t{} = {}".format(measure, score))

    if args.measure == 'bss':

        print("BSS")

        print("\tAfter separation")

        _bss = sep_eval.bss_eval(np.array([enh1, enh2]), np.array([ref1, ref2]))[0]  # (sdr, sir, sar, perm)

        print("\t\tSDR = {}".format(_bss[0]))

        print("\t\tSIR = {}".format(_bss[1]))

        print("\t\tSAR = {}".format(_bss[2]))

        print("\t\tPermutation = {}".format(_bss[3]))

    if args.measure == 'stoi':

        print("STOI")

        print("\tBefore separation")

        _stoi_deg = sep_eval.stoi(deg, ref1)

        print("\t\tSTOI = {}".format(_stoi_deg))

        print("\tAfter separation")

        _stoi_enh = sep_eval.stoi(enh1, ref1)

        print("\t\tSTOI = {}".format(_stoi_enh))
        
    if args.measure == 'pesq':

        print("PESQ")

        print("\tBefore separation")

        _pesq_deg = sep_eval.pesq(deg, ref1)

        print("\t\tPESQ = {}".format(_pesq_deg))

        print("\tAfter separation")

        _pesq_enh = sep_eval.pesq(enh1, ref1)

        print("\t\tPESQ = {}".format(_pesq_enh))
        
    if args.measure == 'sdr':
        print("SDR")

        print("\tBefore separation")

        _sdr_deg = sep_eval.sdr(deg, ref1)

        print("\t\tSDR = {}".format(_sdr_deg))

        print("\tAfter separation")

        _sdr_enh = sep_eval.sdr(enh1, ref1)

        print("\t\tSDR = {}".format(_sdr_enh))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example1')
    parser.add_argument('--measure', type=str, default='full',
                        help='what measure should I test for')
    args, _ = parser.parse_known_args()
    main()

