from poisson_line_process import run
import lhsmdu
import pandas as pd
import multiprocessing
import math
from scipy import stats
import unittest
import os

def run_row(i_row):
    _, row = i_row
    return run(row['width'], row['height'], row['intensity'])

class Tests(unittest.TestCase):
    def test_pp(self):
        # samples = pd.DataFrame(lhsmdu.sample(3, 20).T, columns=["width", "height", "intensity"])
        # samples *= 10
        # samples.to_csv("samples.csv")
        if os.path.exists("samples.csv"):
            samples = pd.read_csv("samples.csv")
        w = samples['width']
        h = samples['height']
        i = samples['intensity']

        pool = multiprocessing.Pool(4)
        # num_lines, num_shapes = zip(*pool.map(run_row, samples.iterrows()))
        num_lines, num_shapes = zip(*[run_row(row) for row in samples.iterrows()])

        samples["num_lines"] = num_lines
        samples["num_shapes"] = num_shapes

        # $L_t=2i(w+h)$
        samples["expected_num_lines"] = 2 * i * (w + h)

        # $P_t=\pi i^2 wh$
        samples["expected_num_shapes"] = math.pi * i * i * w * h

        self.assertTrue(stats.ks_2samp(samples['expected_num_lines'], samples['num_lines']).pvalue > 0.99)
        self.assertTrue(stats.ks_2samp(samples['expected_num_shapes'], samples['num_shapes']).pvalue > 0.99)
