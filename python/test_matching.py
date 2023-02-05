"""Unit tests for maximum weight matching."""

import unittest

from max_weight_matching import maximum_weight_matching as mwm


class TestMaximumWeightMatching(unittest.TestCase):

    def test10_empty(self):
        """empty input graph"""
        self.assertEqual(mwm([]), [])

    def test11_singleedge(self):
        """single edge"""
        self.assertEqual(mwm([(0,1,1)]), [(0,1)])

    def test12(self):
        self.assertEqual(mwm([(1,2,10), (2,3,11)]), [(2,3)])

    def test13(self):
        self.assertEqual(
            mwm([(1,2,5), (2,3,11), (3,4,5)]),
            [(2,3)])

    def test15_float(self):
        """floating point weigths"""
        self.assertEqual(
            mwm([(1,2,3.1415), (2,3,2.7183), (1,3,3.0), (1,4,1.4142)]),
            [(2,3), (1,4)])

    def test16_negative(self):
        """negative weights"""
        self.assertEqual(
            mwm([(1,2,2), (1,3,-2), (2,3,1), (2,4,-1), (3,4,-6)]),
            [(1,2)])

    def test20_sblossom(self):
        """create S-blossom and use it for augmentation"""
        self.assertEqual(
            mwm([(1,2,8), (1,3,9), (2,3,10), (3,4,7)]),
            [(1,2), (3,4)])

    def test20a_sblossom(self):
        """create S-blossom and use it for augmentation"""
        self.assertEqual(
            mwm([(1,2,8), (1,3,9), (2,3,10), (3,4,7), (1,6,5), (4,5,6)]),
            [(2,3), (1,6), (4,5)])

    def test21_tblossom(self):
        """create S-blossom, relabel as T-blossom, use for augmentation"""
        self.assertEqual(
            mwm([(1,2,9), (1,3,8), (2,3,10), (1,4,5), (4,5,4), (1,6,3)]),
            [(2,3),(4,5),(1,6)])

    def test21a_tblossom(self):
        """create S-blossom, relabel as T-blossom, use for augmentation"""
        self.assertEqual(
            mwm([(1,2,9), (1,3,8), (2,3,10), (1,4,5), (4,5,3), (1,6,4)]),
            [(2,3), (4,5), (1,6)])

    def test21b_tblossom(self):
        """create S-blossom, relabel as T-blossom, use for augmentation"""
        self.assertEqual(
            mwm([(1,2,9), (1,3,8), (2,3,10), (1,4,5), (4,5,3), (3,6,4)]),
            [(1,2), (4,5), (3,6)])

    def test22_s_nest(self):
        """create nested S-blossom, use for augmentation"""
        self.assertEqual(
            mwm([(1,2,9), (1,3,9), (2,3,10), (2,4,8), (3,5,8), (4,5,10), (5,6,6)]),
            [(1,3), (2,4), (5,6)])

    def test23_s_relabel_nest(self):
        """create S-blossom, relabel as S, include in nested S-blossom"""
        self.assertEqual(
            mwm([(1,2,10), (1,7,10), (2,3,12), (3,4,20), (3,5,20), (4,5,25), (5,6,10), (6,7,10), (7,8,8)]),
            [(1,2), (3,4), (5,6), (7,8)])

    def test24_s_nest_expand(self):
        """create nested S-blossom, augment, expand recursively"""
        self.assertEqual(
            mwm([(1,2,8), (1,3,8), (2,3,10), (2,4,12), (3,5,12), (4,5,14), (4,6,12), (5,7,12), (6,7,14), (7,8,12)]),
            [(1,2), (3,5), (4,6), (7,8)])

    def test25_s_t_expand(self):
        """create S-blossom, relabel as T, expand"""
        self.assertEqual(
            mwm([(1,2,23), (1,5,22), (1,6,15), (2,3,25), (3,4,22), (4,5,25), (4,8,14), (5,7,13)]),
            [(1,6), (2,3), (4,8), (5,7)])

    def test26_s_nest_t_expand(self):
        """create nested S-blossom, relabel as T, expand"""
        self.assertEqual(
            mwm([(1,2,19), (1,3,20), (1,8,8), (2,3,25), (2,4,18), (3,5,18), (4,5,13), (4,7,7), (5,6,7)]),
            [(1,8), (2,3), (4,7), (5,6)])

    def test30_tnasty_expand(self):
        """create blossom, relabel as T in more than one way, expand, augment"""
        self.assertEqual(
            mwm([(1,2,45), (1,5,45), (2,3,50), (3,4,45), (4,5,50), (1,6,30), (3,9,35), (4,8,35), (5,7,26), (9,10,5)]),
            [(2,3), (1,6), (4,8), (5,7), (9,10)])

    def test31_tnasty2_expand(self):
        """again but slightly different"""
        self.assertEqual(
            mwm([(1,2,45), (1,5,45), (2,3,50), (3,4,45), (4,5,50), (1,6,30), (3,9,35), (4,8,26), (5,7,40), (9,10,5)]),
            [(2,3), (1,6), (4,8), (5,7), (9,10)])

    def test32_t_expand_leastslack(self):
        """create blossom, relabel as T, expand such that a new least-slack S-to-free edge is produced, augment"""
        self.assertEqual(
            mwm([(1,2,45), (1,5,45), (2,3,50), (3,4,45), (4,5,50), (1,6,30), (3,9,35), (4,8,28), (5,7,26), (9,10,5)]),
            [(2,3), (1,6), (4,8), (5,7), (9,10)])

    def test33_nest_tnasty_expand(self):
        """create nested blossom, relabel as T in more than one way, expand outer blossom such that inner blossom ends up on an augmenting path"""
        self.assertEqual(
            mwm([(1,2,45), (1,7,45), (2,3,50), (3,4,45), (4,5,95), (4,6,94), (5,6,94), (6,7,50), (1,8,30), (3,11,35), (5,9,36), (7,10,26), (11,12,5)]),
            [(2,3), (4,6), (1,8), (5,9), (7,10), (11,12)])

    def test34_nest_relabel_expand(self):
        """create nested S-blossom, relabel as S, expand recursively"""
        self.assertEqual(
            mwm([(1,2,40), (1,3,40), (2,3,60), (2,4,55), (3,5,55), (4,5,50), (1,8,15), (5,7,30), (7,6,10), (8,10,10), (4,9,30)]),
            [(1,2), (3,5), (7,6), (8,10), (4,9)])


if __name__ == "__main__":
    unittest.main()

