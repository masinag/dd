"""Utilities for transferring BDDs."""
# Copyright 2016-2018 by California Institute of Technology
# All rights reserved. Licensed under 3-clause BSD.
#
import json
import os
import shelve
import shutil


SHELVE_DIR = '__shelve__'


def copy_vars(source, target):
    """Copy variables, preserving levels.

    @type source, target: `autoref.BDD`
    """
    for var in source.vars:
        level = source.level_of_var(var)
        target.add_var(var, level=level)


def copy_bdds_from(roots, target):
    """Copy BDDs in `roots` to manager `target`.

    @param target: BDD
    """
    cache = dict()
    return [
        copy_bdd(u, target, cache)
        for u in roots]


def copy_bdd(root, target, cache=None):
    """Copy BDD with `root` to manager `target`.

    @param target: BDD or ZDD
    @param cache: `dict` for memoizing results
    """
    if cache is None:
        cache = dict()
    return _copy_bdd(root, target, cache)


def _copy_bdd(u, bdd, cache):
    """Recurse to copy node `u`` to `bdd`.

    @type cache: `dict`
    """
    # terminal ?
    if u == u.bdd.true:
        return bdd.true
    # could be handled via cache,
    # but frequent case
    if u == u.bdd.false:
        return bdd.false
    # rectify
    z = _flip(u, u)
    # non-terminal
    # memoized ?
    k = int(z)
    if k in cache:
        r = cache[k]
        return _flip(r, u)
    # recurse
    low = _copy_bdd(u.low, bdd, cache)
    high = _copy_bdd(u.high, bdd, cache)
    # canonicity
    # if low.negated != u.low.negated:
    #     raise AssertionError((low, u.low))
    # if high.negated:
    #     raise AssertionError(high)
    # add node
    g = bdd.var(u.var)
    r = bdd.ite(g, high, low)
    # if r.negated:
    #     raise AssertionError(r)
    # memoize
    cache[k] = r
    return _flip(r, u)


def _flip(r, u):
    """Negate `r` if `u` is negated.

    Else return `r`.
    """
    return ~ r if u.negated else r


def copy_zdd(root, target, cache=None):
    """Copy ZDD with `root` to manager `target`.

    @param target: BDD or ZDD
    @param cache: `dict` for memoizing results
    """
    if cache is None:
        cache = dict()
    level = 0
    return _copy_zdd(level, root, target, cache)


def _copy_zdd(level, u, target, cache):
    """Recurse to copy node `u` to `target`.

    @type cache: `dict`
    """
    src = u.bdd
    # terminal ?
    if u == src.false:
        return target.false
    if level == len(src.vars):
        return target.true
    # memoized ?
    k = int(u)
    if k in cache:
        return cache[k]
    # recurse
    v, w = src._top_cofactor(u, level)
    low = _copy_zdd(
        level + 1, v, target, cache)
    high = _copy_zdd(
        level + 1, w, target, cache)
    # add node
    var = src.var_at_level(level)
    g = target.var(var)
    r = target.ite(g, high, low)
    # memoize
    cache[k] = r
    return r


def dump_json(nodes, file_name):
    """Write reachable nodes to JSON file.

    Writes the nodes that are reachable from
    the roots in `nodes` to the JSON file
    named `file_name`.

    Also dumps the variable names and the
    variable order, to the same JSON file.
    """
    tmp_fname = os.path.join(
        SHELVE_DIR, 'temporary_shelf')
    os.makedirs(SHELVE_DIR)
    try:
        with shelve.open(tmp_fname) as cache,\
                open(file_name, 'w') as fd:
            _dump_json(nodes, fd, cache)
    finally:
        # `shelve` file naming
        # depends on context
        shutil.rmtree(SHELVE_DIR)


def _dump_json(nodes, fd, cache):
    """Dump BDD as JSON to file `fd`.

    Use `cache` to keep track of
    visited nodes.
    """
    fd.write('{')
    _dump_bdd_info(nodes, fd)
    for u in nodes:
        _dump_bdd(u, fd, cache)
    fd.write('\n}\n')


def _dump_bdd_info(nodes, fd):
    """Dump variable levels and roots."""
    u = next(iter(nodes))
    bdd = u.bdd
    var_level = {
        var: bdd.level_of_var(var)
        for var in bdd.vars}
    roots = [_node_to_int(u) for u in nodes]
    info = (
        '\n"level_of_var": {level}'
        ',\n"roots": {roots}').format(
            level=json.dumps(var_level),
            roots=json.dumps(roots))
    fd.write(info)


def _dump_bdd(u, fd, cache):
    """Recursive step of dumping nodes."""
    # terminal ?
    if u == u.bdd.true:
        return '"T"'
    if u == u.bdd.false:
        return '"F"'
    # rectify
    z = _flip(u, u)
    # non-terminal
    # dumped ?
    k = int(z)
    if str(k) in cache:
        return -k if u.negated else k
    # recurse
    low = _dump_bdd(u.low, fd, cache)
    high = _dump_bdd(u.high, fd, cache)
    # dump node
    s = f',\n"{k}": [{u.level}, {low}, {high}]'
    fd.write(s)
    # record as dumped
    cache[str(k)] = True
    return -k if u.negated else k


def load_json(file_name, bdd, load_order=False):
    """Add BDDs from JSON `file_name` to `bdd`.

    @param load_order: if `True`,
        then load variable order
        from `file_name`.
    """
    tmp_fname = os.path.join(
        SHELVE_DIR, 'temporary_shelf')
    os.makedirs(SHELVE_DIR)
    try:
        with shelve.open(tmp_fname) as cache,\
                open(file_name, 'r') as fd:
            nodes = _load_json(
                fd, bdd, load_order, cache)
    finally:
        shutil.rmtree(SHELVE_DIR)
    return nodes


def _load_json(fd, bdd, load_order, cache):
    """Load BDDs from JSON file `fd` to manager `bdd`."""
    context = dict(load_order=load_order)
    # if the variable order is going to be loaded,
    # then turn off dynamic reordering,
    # because it can change the order midway,
    # which would not return the loaded order,
    # and can also cause failure of
    # the assertion below
    if load_order:
        old_reordering = bdd.configure(
            reordering=False)
    for line in fd:
        d = _parse_line(line)
        _store_line(d, bdd, context, cache)
    roots = [_node_from_int(k, bdd, cache)
             for k in context['roots']]
    # rm refs to cached nodes
    for uid in cache:
        u = _node_from_int(int(uid), bdd, cache)
        if u.ref < 2:
            raise AssertionError(u.ref)
            # +1 ref due to `incref` in `_make_node`
            # +1 ref due to the `_node_from_int`
            #   call for `u`
        if load_order and u.ref < 3:
            raise AssertionError(u.ref)
            # +1 ref due to `incref` in `_make_node`
            # +1 ref due to either:
            #   - being a successor node
            #   - being a root node
            #     (thus referenced in `roots` above)
            # +1 ref due to the `_node_from_int`
            #   call for `u`
        bdd.decref(u, _direct=True)
            # this module is unusual,
            # in that `incref` and `decref` need
            # to be called on different `Function`
            # instances for the same node
    bdd.assert_consistent()
    if load_order:
        bdd.configure(
            reordering=old_reordering)
    return roots


def _parse_line(line):
    """Parse JSON from `line`."""
    line = line.rstrip()
    if line == '{' or line == '}':
        return
    if line.endswith(','):
        line = line.rstrip(',')
    return json.loads('{' + line + '}')


def _store_line(d, bdd, context, cache):
    """Interpret data in `d`."""
    if d is None:
        return
    order = d.get('level_of_var')
    if order is not None:
        order = {
            str(k): v
            for k, v in order.items()}
        bdd.declare(*order)
        context['level_of_var'] = order
        context['var_at_level'] = {
            v: k for k, v in order.items()}
        if context['load_order']:
            bdd.reorder(order)
        return
    roots = d.get('roots')
    if roots is not None:
        context['roots'] = roots
        return
    _make_node(d, bdd, context, cache)


def _make_node(d, bdd, context, cache):
    """Create a new node in `bdd` from `d`."""
    (uid, (level, low_id, high_id)), = d.items()
    k, level = map(int, (uid, level))
    if k <= 0:
        raise AssertionError(k)
    if level < 0:
        raise AssertionError(level)
    low_id = _decode_node(low_id)
    high_id = _decode_node(high_id)
    if str(k) in cache:
        return
    low = _node_from_int(low_id, bdd, cache)
    high = _node_from_int(high_id, bdd, cache)
    var = context['var_at_level'][level]
    if context['load_order']:
        u = bdd.find_or_add(var, low, high)
    else:
        g = bdd.var(var)
        u = bdd.ite(g, high, low)
    if u.negated:
        raise AssertionError(u)
    # memoize
    cache[str(k)] = int(u)
    bdd.incref(u)


def _decode_node(s):
    """Map string `s` to node-like integer."""
    if s == 'F':
        return -1
    elif s == 'T':
        return 1
    else:
        return int(s)


def _node_from_int(uid, bdd, cache):
    """Return `bdd` node `u` from integer `uid`.

    @type uid: `int`
    @type bdd: `BDD`
    @type cache: `dict`-like
    @rtype: `Function`
    """
    if uid == -1:
        return bdd.false
    elif uid == 1:
        return bdd.true
    # not constant
    k = cache[str(abs(uid))]
    u = bdd._add_int(k)
    return ~ u if uid < 0 else u


def _node_to_int(u):
    """Return integer representing node `u`."""
    z = _flip(u, u)
    k = int(z)
    return -k if u.negated else k
