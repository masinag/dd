"""How to use ZDDs with CUDD."""
import dd.ldd as _ldd


def ldd_example():
    ldd = _ldd.LDD(nvars=3)
    # x0 - x2 <= -3
    lddc1 = ldd.constraint(((1, 0, 1), False, -3))
    # -x0 + x1 <= 2
    lddc2 = ldd.constraint(((-1, 1, 0), False, 2))
    # -x1 + x2 <= -1
    lddc3 = ldd.constraint(((0, -1, 1), False, -1))

    ldd_or = (lddc1 | lddc2) | lddc3
    n_nodes = len(ldd_or)
    print(f"Number of nodes ldd_or: {n_nodes}")
    ldd.dump("ldd_or.pdf", [ldd_or])

    ldd_or_lemmas = (lddc1 | lddc2 | lddc3) & (~lddc1 | ~lddc2 | ~lddc3)
    n_nodes = len(ldd_or_lemmas)
    print(f"Number of nodes ldd_or_lemmas: {n_nodes}")
    ldd.dump("ldd_or_lemmas.pdf", [ldd_or_lemmas])

if __name__ == '__main__':
    ldd_example()
