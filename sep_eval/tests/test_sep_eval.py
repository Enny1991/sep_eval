from __future__ import print_function
from unittest import TestCase
import numpy as np
import soundfile as sf
import warnings

from sep_eval.sep_eval import pesq, stoi, sdr, bss_eval

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestSepEval(TestCase):

    def setUp(self):
        deg, fs = sf.read('wavs/degraded.wav')
        self.deg = deg
        self.fs = fs
        ref, fs = sf.read('wavs/reference1.wav')
        self.assertEqual(self.fs, fs)
        self.ref = ref
        enh, fs = sf.read('wavs/enhanced1.wav')
        self.assertEqual(self.fs, fs)
        self.enh = enh
        ref2, fs = sf.read('wavs/reference2.wav')
        self.assertEqual(self.fs, fs)
        self.ref2 = ref2
        enh2, fs = sf.read('wavs/enhanced2.wav')
        self.assertEqual(self.fs, fs)
        self.enh2 = enh2

    # def test_pesq(self):
    #
    #     _pesq_deg = pesq(self.deg, self.ref)
    #     _pesq_enh = pesq(self.enh, self.ref)
    #
    #     self.assertEqual(_pesq_deg, 2.202)
    #     self.assertEqual(_pesq_enh, 2.619)
    #     np.testing.assert_equal([_pesq_deg, _pesq_enh], [2.202,  2.619])
    #
    # def test_pesq_multiple_list(self):
    #
    #     _pesq_enh = pesq([self.enh, self.enh2], [self.ref, self.ref2])
    #
    #     self.assertAlmostEqual(len(_pesq_enh), 2)
    #     np.testing.assert_almost_equal(_pesq_enh, [2.619, 2.648])
    #
    #     _pesq_enh_avg = pesq([self.enh, self.enh2], [self.ref, self.ref2], average=True)
    #
    #     np.testing.assert_almost_equal(_pesq_enh_avg, 2.6335)
    #
    # def test_pesq_multiple_array(self):
    #
    #     _pesq_enh = pesq(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]))
    #
    #     self.assertAlmostEqual(len(_pesq_enh), 2)
    #     np.testing.assert_almost_equal(_pesq_enh, [2.619, 2.648])
    #
    #     _pesq_enh_avg = pesq(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]), average=True)
    #
    #     np.testing.assert_almost_equal(_pesq_enh_avg, 2.6335)

    def test_stoi_single(self):

        _stoi_deg = stoi(self.deg, self.ref)
        _stoi_enh = stoi(self.enh, self.ref)

        self.assertAlmostEqual(_stoi_deg, 0.6609205519722758)
        self.assertAlmostEqual(_stoi_enh, 0.8010709605254842)
        np.testing.assert_almost_equal([_stoi_deg, _stoi_enh], [0.6609205519722758,  0.8010709605254842])

    def test_stoi_multiple_list(self):

        _stoi_enh = stoi([self.enh, self.enh2], [self.ref, self.ref2])

        self.assertAlmostEqual(len(_stoi_enh), 2)
        np.testing.assert_almost_equal(_stoi_enh, [0.8010709605254842, 0.6892444])

        _stoi_enh_avg = stoi([self.enh, self.enh2], [self.ref, self.ref2], average=True)

        np.testing.assert_almost_equal(_stoi_enh_avg, 0.7451576802627421)

    def test_stoi_multiple_array(self):

        _stoi_enh = stoi(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]))

        self.assertAlmostEqual(len(_stoi_enh), 2)
        np.testing.assert_almost_equal(_stoi_enh, [0.8010709605254842, 0.6892444])

        _stoi_enh_avg = stoi(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]), average=True)

        np.testing.assert_almost_equal(_stoi_enh_avg, 0.7451576802627421)

    def test_stoi_extended(self):

        _stoi_deg = stoi(self.deg, self.ref, extended=True)
        _stoi_enh = stoi(self.enh, self.ref, extended=True)

        self.assertAlmostEqual(_stoi_deg, 0.6071910004054383)
        self.assertAlmostEqual(_stoi_enh, 0.710870118264931)
        np.testing.assert_almost_equal([_stoi_deg, _stoi_enh], [0.6071910004054383, 0.710870118264931])

    def test_sdr_single(self):

        _sdr_deg = sdr(self.deg, self.ref)
        _sdr_enh = sdr(self.enh, self.ref)

        np.testing.assert_equal([_sdr_deg, _sdr_enh], [2.827279048726137,  4.362387694890646])
        
    def test_sdr_multiple_list(self):

        _sdr_enh = sdr([self.enh, self.enh2], [self.ref, self.ref2])

        self.assertAlmostEqual(len(_sdr_enh), 2)
        np.testing.assert_almost_equal(_sdr_enh, [4.362387694890646, 2.92916932])

        _sdr_enh_avg = sdr([self.enh, self.enh2], [self.ref, self.ref2], average=True)

        np.testing.assert_almost_equal(_sdr_enh_avg, 3.645778507445323)

    def test_sdr_multiple_array(self):

        _sdr_enh = sdr(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]))

        self.assertAlmostEqual(len(_sdr_enh), 2)
        np.testing.assert_almost_equal(_sdr_enh, [4.362387694890646, 2.92916932])

        _sdr_enh_avg = sdr(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]), average=True)
        np.testing.assert_almost_equal(_sdr_enh_avg, 3.645778507445323)

    def test_bss(self):

        _bss = bss_eval(np.array([self.enh, self.enh2]), np.array([self.ref, self.ref2]))

        np.testing.assert_almost_equal(_bss['sdr'], np.array([8.329875273583793, 6.51470304]))
        np.testing.assert_almost_equal(_bss['sir'], np.array([18.19676815, 14.14325955]))
        np.testing.assert_almost_equal(_bss['sar'], np.array([8.86778371, 7.50190804]))
        np.testing.assert_almost_equal(_bss['perm'], np.array([0, 1]))

    def test_bss_multiple(self):

        deg = [np.array([self.enh, self.enh2]),
               np.array([self.enh2, self.enh])]

        ref = [np.array([self.ref, self.ref2]),
               np.array([self.ref2, self.ref])]

        _bss = bss_eval(deg, ref)

        _true_sdr = np.array([np.array([8.329875273583793, 6.51470304]),
                              np.array([6.51470304, 8.329875273583793])])

        _true_sir = np.array([np.array([18.19676815, 14.14325955]),
                              np.array([14.14325955, 18.19676815])])

        _true_sar = np.array([np.array([8.86778371, 7.50190804]),
                              np.array([7.50190804, 8.86778371])])

        _true_perm = np.array([np.array([0, 1]),
                              np.array([0, 1])])

        np.testing.assert_almost_equal(_bss['sdr'], _true_sdr)
        np.testing.assert_almost_equal(_bss['sir'], _true_sir)
        np.testing.assert_almost_equal(_bss['sar'], _true_sar)
        np.testing.assert_almost_equal(_bss['perm'], _true_perm)

        _bss_avg = bss_eval(deg, ref, average=True)

        np.testing.assert_almost_equal(_bss_avg['sdr'], np.mean(_true_sdr, 0))
        np.testing.assert_almost_equal(_bss_avg['sir'], np.mean(_true_sir, 0))
        np.testing.assert_almost_equal(_bss_avg['sar'], np.mean(_true_sar, 0))
        np.testing.assert_almost_equal(_bss_avg['perm'], _true_perm)
