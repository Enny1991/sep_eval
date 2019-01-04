"""
You can use the measures also in batch mode, just pass a list or an array of degrade and reference signals
"""
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

        print("\tAfter separation")
        # you can use arrays ...
        eval_enh1 = sep_eval.full_eval([enh1, enh2], [ref1, ref2])
        # or lists
        eval_enh1 = sep_eval.full_eval(np.array([enh1, enh2]), np.array([ref1, ref2]))

        for measure, score in eval_enh1.items():
            print("\t\t{} = {}".format(measure, score))

        print("\tAVERAGE")
        # You can also get directly the average
        # you can use arrays ...
        eval_enh1 = sep_eval.full_eval([enh1, enh2], [ref1, ref2], average=True)
        # or lists
        eval_enh1 = sep_eval.full_eval(np.array([enh1, enh2]), np.array([ref1, ref2]), average=True)

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
        # you can use arrays ...
        _stoi_deg = sep_eval.stoi(np.array([deg, deg]), np.array([ref1, ref2]))
        # or lists ...
        _stoi_deg = sep_eval.stoi([deg, deg], [ref1, ref2])

        print("\t\tSTOI = {}".format(_stoi_deg))

        print("\tAfter separation")
        # you can use arrays ...
        _stoi_enh = sep_eval.stoi(np.array([enh1, enh2]), np.array([ref1, ref2]))
        # or lists ...
        _stoi_enh = sep_eval.stoi([enh1, enh2], [ref1, ref2])

        print("\t\tSTOI = {}".format(_stoi_enh))

        print("\tAVERAGE")
        print("\tAfter separation")
        # you can use arrays ...
        _stoi_enh = sep_eval.stoi(np.array([enh1, enh2]), np.array([ref1, ref2]), average=True)
        # or lists ...
        _stoi_enh = sep_eval.stoi([enh1, enh2], [ref1, ref2], average=True)

        print("\t\tSTOI = {}".format(_stoi_enh))
        
    if args.measure == 'sdr':

        print("SDR")

        print("\tBefore separation")
        # you can use arrays ...
        _sdr_deg = sep_eval.sdr(np.array([deg, deg]), np.array([ref1, ref2]))
        # or lists ...
        _sdr_deg = sep_eval.sdr([deg, deg], [ref1, ref2])

        print("\t\tSDR = {}".format(_sdr_deg))

        print("\tAfter separation")
        # you can use arrays ...
        _sdr_enh = sep_eval.sdr(np.array([enh1, enh2]), np.array([ref1, ref2]))
        # or lists ...
        _sdr_enh = sep_eval.sdr([enh1, enh2], [ref1, ref2])

        print("\t\tSDR = {}".format(_sdr_enh))

        print("\tAVERAGE")
        print("\tAfter separation")
        # you can use arrays ...
        _sdr_enh = sep_eval.sdr(np.array([enh1, enh2]), np.array([ref1, ref2]), average=True)
        # or lists ...
        _sdr_enh = sep_eval.sdr([enh1, enh2], [ref1, ref2], average=True)

        print("\t\tSDR = {}".format(_sdr_enh))
        
    if args.measure == 'pesq':

        print("PESQ")

        print("\tBefore separation")
        # you can use arrays ...
        _pesq_deg = sep_eval.pesq(np.array([deg, deg]), np.array([ref1, ref2]))
        # or lists ...
        _pesq_deg = sep_eval.pesq([deg, deg], [ref1, ref2])

        print("\t\tPESQ = {}".format(_pesq_deg))

        print("\tAfter separation")
        # you can use arrays ...
        _pesq_enh = sep_eval.pesq(np.array([enh1, enh2]), np.array([ref1, ref2]))
        # or lists ...
        _pesq_enh = sep_eval.pesq([enh1, enh2], [ref1, ref2])

        print("\t\tPESQ = {}".format(_pesq_enh))

        print("\tAVERAGE")
        print("\tAfter separation")
        # you can use arrays ...
        _pesq_enh = sep_eval.pesq(np.array([enh1, enh2]), np.array([ref1, ref2]), average=True)
        # or lists ...
        _pesq_enh = sep_eval.pesq([enh1, enh2], [ref1, ref2], average=True)

        print("\t\tPESQ = {}".format(_pesq_enh))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example1')
    parser.add_argument('--measure', type=str, default='full',
                        help='what measure should I test for')
    args, _ = parser.parse_known_args()
    main()

