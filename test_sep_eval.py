from __future__ import print_function
from unittest import TestCase
import pkg_resources
import logging
import numpy as np
import soundfile as sf
import warnings

from sep_eval import pesq, stoi, sdr, bss_eval

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestDeepExplainGeneralTF(TestCase):

    def setUp(self):
        print("starting")

    def tearDown(self):
        print("done")

    def test_pesq(self):
        deg, fs = sf.read('wavs/degraded.wav')
        ref, fs = sf.read('wavs/reference1.wav')
        enh, fs = sf.read('wavs/enhanced1.wav')

        _pesq_deg = pesq(deg, ref)
        _pesq_enh = pesq(enh, ref)

        self.assertEqual(_pesq_deg, 2.202)
        self.assertEqual(_pesq_enh, 2.619)
        np.testing.assert_equal([_pesq_deg, _pesq_enh], [2.202,  2.619])

    def test_stoi(self):
        deg, fs = sf.read('wavs/degraded.wav')
        ref, fs = sf.read('wavs/reference1.wav')
        enh, fs = sf.read('wavs/enhanced1.wav')

        # normal
        _stoi_deg = stoi(deg, ref)
        _stoi_enh = stoi(enh, ref)

        self.assertAlmostEqual(_stoi_deg, 0.6609205519722758)
        self.assertAlmostEqual(_stoi_enh, 0.8010709605254842)
        np.testing.assert_almost_equal([_stoi_deg, _stoi_enh], [0.6609205519722758,  0.8010709605254842])

        # extended
        _stoi_deg = stoi(deg, ref, extended=True)
        _stoi_enh = stoi(enh, ref, extended=True)

        self.assertAlmostEqual(_stoi_deg, 0.6071910004054383)
        self.assertAlmostEqual(_stoi_enh, 0.710870118264931)
        np.testing.assert_almost_equal([_stoi_deg, _stoi_enh], [0.6071910004054383, 0.710870118264931])

    def test_sdr(self):
        deg, fs = sf.read('wavs/degraded.wav')
        ref, fs = sf.read('wavs/reference1.wav')
        enh, fs = sf.read('wavs/enhanced1.wav')

        _sdr_deg = sdr(deg, ref)
        _sdr_enh = sdr(enh, ref)

        self.assertEqual(_sdr_deg, 2.827279048726137)
        self.assertEqual(_sdr_enh, 4.362387694890646)
        np.testing.assert_equal([_sdr_deg, _sdr_enh], [2.827279048726137,  4.362387694890646])

    def test_bss(self):
        ref1, fs = sf.read('wavs/reference1.wav')
        enh1, fs = sf.read('wavs/enhanced1.wav')
        ref2, fs = sf.read('wavs/reference2.wav')
        enh2, fs = sf.read('wavs/enhanced2.wav')

        _bss = bss_eval(np.array([enh1, enh2]), np.array([ref1, ref2]))[0]

        self.assertAlmostEqual(_bss[0][0], 8.329875273583793)

        np.testing.assert_almost_equal(_bss, (np.array([8.329875273583793, 6.51470304]),
                                              np.array([18.19676815, 14.14325955]),
                                              np.array([8.86778371, 7.50190804]),
                                              np.array([0, 1])))

