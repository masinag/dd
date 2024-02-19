"""Tests of the module `dd.cudd`."""
# This file is released in the public domain.
#
import logging

import dd.ldd as _ldd

logging.getLogger('astutils').setLevel('ERROR')


def lincons1(ldd):
    coeffs = (1, -3)
    strict = False
    constant = 0
    lincons = (coeffs, strict, constant)
    logging.debug('lincons: %s', lincons)
    lddc = ldd.constraint(lincons)
    return lddc


def lincons2(ldd):
    coeffs = (1, 1)
    strict = False
    constant = 0
    lincons = (coeffs, strict, constant)
    logging.debug('lincons: %s', lincons)
    lddc = ldd.constraint(lincons)
    return lddc


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


def test_decref():
    _ldd._test_decref()


def test_true():
    ldd = _ldd.LDD()
    t1 = ldd.true
    t2 = ldd.true
    assert t1 == t2, (t1, t2)


def test_false():
    ldd = _ldd.LDD()
    f1 = ldd.false
    f2 = ldd.false
    assert f1 == f2, (f1, f2)


def test_lincons():
    # enable logging
    ldd = _ldd.LDD()
    # e.g., x - 3 * y <= 0
    coeffs = (1, -3)
    strict = False
    constant = 0
    lincons = (coeffs, strict, constant)
    logging.debug('lincons: %s', lincons)
    lddc = ldd.constraint(lincons)


def test_and():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2


def test_high_low():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    h = lddc.high
    l = lddc.low
    assert h == lddc1 or h == lddc2, h
    assert l == ldd.false, l


def test_index():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    i = lddc1._index
    assert 0 == i, i
    j = lddc2._index
    assert 1 == j, j


def test_level():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    assert 0 == lddc.level, lddc.level
    assert 1 == lddc.high.level


def test_var():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    assert ldd.var("x0-3*x1<=0") == lddc1
    assert ldd.var("x0+x1<=0") == lddc2


def test_var_at_level():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    assert ldd.var_at_level(0) == lddc1.var
    assert ldd.var_at_level(1) == lddc2.var


def test_count_nodes():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    n = _ldd.count_nodes([lddc])
    assert n == 3, n



def test_to_expr():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    e = ldd.to_expr(lddc)
    assert e == "ite(x0-3*x1<=0, ite(x0+x1<=0, FALSE, TRUE), TRUE)", e


def test_dump():
    ldd = _ldd.LDD()
    lddc1 = lincons1(ldd)
    lddc2 = lincons2(ldd)
    lddc = lddc1 & lddc2
    ldd.dump("dump.pdf", [lddc])
    ldd.dump("dump.json", [lddc])
