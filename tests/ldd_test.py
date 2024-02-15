"""Tests of the module `dd.cudd`."""
# This file is released in the public domain.
#
import logging

import dd.ldd as _ldd
import pytest

logging.getLogger('astutils').setLevel('ERROR')


def test_init():
    ldd = _ldd.LDD(_ldd.UTVPIZ)


def test_len():
    ldd = _ldd.LDD()
    assert len(ldd) == 0, len(ldd)

def test_str():
    ldd = _ldd.LDD()
    s = str(ldd)

def test_incref():
    _ldd._test_incref()

