#!/usr/bin/env python
"""
simple example script for running notebooks and reporting exceptions.
Usage: `checkipnb.py foo.ipynb [bar.ipynb [...]]`
Each cell is submitted to the kernel, and checked for errors.
"""

import os
import glob
from runipy.notebook_runner import NotebookRunner

from pyfolio.utils import pyfolio_root
from pyfolio.ipycompat import read as read_notebook


def test_nbs():
    path = os.path.join(pyfolio_root(), 'examples', '*.ipynb')
    for ipynb in glob.glob(path):
        with open(ipynb) as f:
            nb = read_notebook(f, 'json')
            nb_runner = NotebookRunner(nb)
            nb_runner.run_notebook(skip_exceptions=False)
