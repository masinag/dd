"""How to use ZDDs with CUDD."""
import dd.ldd as _ldd


def ldd_example():
    ldd = _ldd.LDD()
    n_nodes = len(ldd)
    print("Number of nodes in the LDD: ", n_nodes)
    # zdd.declare('x', 'y', 'z')
    # u = zdd.add_expr(r'(x /\ y) \/ z')
    # let = dict(y=zdd.add_expr('~ x'))
    # v = zdd.let(let, u)
    # v_ = zdd.add_expr('z')
    # assert v == v_, (v, v_)


if __name__ == '__main__':
    ldd_example()
