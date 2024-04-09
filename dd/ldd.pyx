"""Cython interface to LDD.

Reference
=========
    Fabio Somenzi
    "CUDD: CU Decision Diagram Package"
    University of Colorado at Boulder
    v2.5.1, 2015
    <http://vlsi.colorado.edu/~fabio/>

    S. Chaki, A. Gurfinkel, and O. Strichman,
    "Decision diagrams for linear arithmetic"
    FMCAD 2009
"""
import collections.abc as _abc
import logging
import textwrap as _tw
import typing as _ty

cimport libc.stdint as stdint
from cpython cimport bool as python_bool
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdio cimport FILE, fseek, SEEK_SET, fclose, fread, fopen
from libcpp cimport bool

import dd._abc as _dd_abc
from dd import _utils, _copy, autoref

_Yes: _ty.TypeAlias = python_bool
_Nat: _ty.TypeAlias = _dd_abc.Nat
_Cardinality: _ty.TypeAlias = _dd_abc.Cardinality
_NumberOfBytes: _ty.TypeAlias = _dd_abc.NumberOfBytes
_VariableName: _ty.TypeAlias = _dd_abc.VariableName
_LinearTerm: _ty.TypeAlias = tuple[int, ...]
_VariableName: _ty.TypeAlias = _dd_abc.VariableName
_Constant: _ty.TypeAlias = int | float | tuple[int, int]
_LinearConstraint: _ty.TypeAlias = tuple[_LinearTerm, _Yes, _Constant]
_Level: _ty.TypeAlias = _dd_abc.Level
_VariableLevels: _ty.TypeAlias = _dd_abc.VariableLevels
_Assignment: _ty.TypeAlias = dict[_LinearConstraint, python_bool]
_Renaming: _ty.TypeAlias = _dd_abc.Renaming
_Formula: _ty.TypeAlias = _dd_abc.Formula
_BDDFileType: _ty.TypeAlias = (_dd_abc.BDDFileType | _ty.Literal['dddmp'])

# -------------------- CUDD --------------------
# Copyright 2015 by California Institute of Technology
# All rights reserved. Licensed under BSD-3.
#

cdef extern from 'mtr.h':
    struct MtrNode_:
        pass
    ctypedef MtrNode_ MtrNode

cdef extern from 'cuddInt.h':
    char * CUDD_VERSION
    int CUDD_CONST_INDEX
    # subtable (for a level)
    struct DdSubtable:
        unsigned int slots
        unsigned int keys
    # manager
    struct DdManager:
        DdSubtable *subtables
        unsigned int keys
        unsigned int dead
        double cachecollisions
        double cacheinserts
        double cachedeletions
    DdNode *cuddUniqueInter(
            DdManager *unique,
            int index,
            DdNode *T, DdNode *E)
cdef extern from 'cudd.h':
    # node
    ctypedef unsigned int DdHalfWord
    struct DdNode:
        DdHalfWord index
        DdHalfWord ref
    ctypedef DdNode DdNode

    ctypedef DdManager DdManager
    DdManager *Cudd_Init(
            unsigned int numVars,
            unsigned int numVarsZ,
            unsigned int numSlots,
            unsigned int cacheSize,
            size_t maxMemory)
    struct DdGen
    ctypedef enum Cudd_ReorderingType:
        pass
    # node elements
    DdNode *Cudd_bddNewVar(
            DdManager *dd)
    DdNode *Cudd_bddNewVarAtLevel(
            DdManager *dd, int level)
    DdNode *Cudd_bddIthVar(
            DdManager *dd, int index)
    DdNode *Cudd_ReadLogicZero(
            DdManager *dd)
    DdNode *Cudd_ReadOne(
            DdManager *dd)
    DdNode *Cudd_Regular(
            DdNode *u)
    bool Cudd_IsConstant(
            DdNode *u)
    unsigned int Cudd_NodeReadIndex(
            DdNode *u)
    DdNode *Cudd_T(
            DdNode *u)
    DdNode *Cudd_E(
            DdNode *u)
    bool Cudd_IsComplement(
            DdNode *u)
    void Cudd_IterDerefBdd(
            DdManager * table,
            DdNode * n
    )
    int Cudd_DagSize(
            DdNode *node)
    int Cudd_SharingSize(
            DdNode ** nodeArray, int n)
    # basic Boolean operators
    DdNode *Cudd_Not(
            DdNode *dd)
    DdNode *Cudd_bddIte(
            DdManager *dd, DdNode *f,
            DdNode *g, DdNode *h)
    DdNode *Cudd_bddAnd(
            DdManager *dd,
            DdNode *f, DdNode *g)
    DdNode *Cudd_bddOr(
            DdManager *dd,
            DdNode *f, DdNode *g)
    DdNode *Cudd_bddXor(
            DdManager *dd,
            DdNode *f, DdNode *g)
    DdNode *Cudd_bddXnor(
            DdManager *dd,
            DdNode *f, DdNode *g)
    DdNode *Cudd_Support(
            DdManager *dd, DdNode *f)
    DdNode *Cudd_bddComputeCube(
            DdManager *dd,
            DdNode ** vars, int *phase, int n)
    DdNode *Cudd_CubeArrayToBdd(
            DdManager *dd, int *array)
    int Cudd_BddToCubeArray(
            DdManager *dd,
            DdNode *cube, int *array)
    int Cudd_PrintMinterm(
            DdManager *dd, DdNode *f)
    int Cudd_PrintDebug(
            DdManager * dd,
            DdNode * f,
            int  n,
            int  pr
    )
    DdNode *Cudd_Cofactor(
            DdManager *dd, DdNode *f, DdNode *g)
    DdNode *Cudd_bddCompose(
            DdManager *dd,
            DdNode *f, DdNode *g, int v)
    DdNode *Cudd_bddVectorCompose(
            DdManager *dd,
            DdNode *f, DdNode ** vector)
    DdNode *Cudd_bddRestrict(
            DdManager *dd, DdNode *f, DdNode *c)
    # cubes
    DdGen *Cudd_FirstCube(
            DdManager *dd, DdNode *f,
            int ** cube, double *value)
    int Cudd_NextCube(
            DdGen *gen, int ** cube, double *value)
    int Cudd_IsGenEmpty(
            DdGen *gen)
    int Cudd_GenFree(
            DdGen *gen)
    double Cudd_CountMinterm(
            DdManager *dd, DdNode *f, int nvars)
    # refs
    void Cudd_Ref(
            DdNode *n)
    void Cudd_RecursiveDeref(
            DdManager *table, DdNode *n)
    void Cudd_Deref(
            DdNode *n)
    # checks
    int Cudd_CheckZeroRef(
            DdManager *manager)
    int Cudd_DebugCheck(
            DdManager *table)
    void Cudd_Quit(
            DdManager *unique)
    DdNode *Cudd_bddTransfer(
            DdManager *ddSource,
            DdManager *ddDestination,
            DdNode *f)
    # info
    int Cudd_PrintInfo(
            DdManager *dd, FILE *fp)
    int Cudd_ReadSize(
            DdManager *dd)
    long Cudd_ReadNodeCount(
            DdManager *dd)
    long Cudd_ReadPeakNodeCount(
            DdManager *dd)
    int Cudd_ReadPeakLiveNodeCount(
            DdManager *dd)
    size_t Cudd_ReadMemoryInUse(
            DdManager *dd)
    unsigned int Cudd_ReadSlots(
            DdManager *dd)
    double Cudd_ReadUsedSlots(
            DdManager *dd)
    double Cudd_ExpectedUsedSlots(
            DdManager *dd)
    unsigned int Cudd_ReadCacheSlots(
            DdManager *dd)
    double Cudd_ReadCacheUsedSlots(
            DdManager *dd)
    double Cudd_ReadCacheLookUps(
            DdManager *dd)
    double Cudd_ReadCacheHits(
            DdManager *dd)
    # reordering
    int Cudd_ReduceHeap(
            DdManager *table,
            Cudd_ReorderingType heuristic,
            int minsize)
    int Cudd_ShuffleHeap(
            DdManager *table, int *permutation)
    void Cudd_AutodynEnable(
            DdManager *unique,
            Cudd_ReorderingType method)
    void Cudd_AutodynDisable(
            DdManager *unique)
    int Cudd_ReorderingStatus(
            DdManager *unique,
            Cudd_ReorderingType *method)
    unsigned int Cudd_ReadReorderings(
            DdManager *dd)
    long Cudd_ReadReorderingTime(
            DdManager *dd)
    int Cudd_ReadPerm(
            DdManager *dd, int index)
    int Cudd_ReadInvPerm(
            DdManager *dd, int level)
    void Cudd_SetSiftMaxSwap(
            DdManager *dd, int sms)
    int Cudd_ReadSiftMaxSwap(
            DdManager *dd)
    void Cudd_SetSiftMaxVar(
            DdManager *dd, int smv)
    int Cudd_ReadSiftMaxVar(
            DdManager *dd)
    # variable grouping
    extern MtrNode *Cudd_MakeTreeNode(
            DdManager *dd, unsigned int low,
            unsigned int size, unsigned int type)
    extern MtrNode *Cudd_ReadTree(
            DdManager *dd)
    extern void Cudd_SetTree(
            DdManager *dd, MtrNode *tree)
    extern void Cudd_FreeTree(
            DdManager *dd)
    # manager config
    size_t Cudd_ReadMaxMemory(
            DdManager *dd)
    size_t Cudd_SetMaxMemory(
            DdManager *dd,
            size_t maxMemory)
    unsigned int Cudd_ReadMaxCacheHard(
            DdManager *dd)
    unsigned int Cudd_ReadMaxCache(
            DdManager *dd)
    void Cudd_SetMaxCacheHard(
            DdManager *dd, unsigned int mc)
    double Cudd_ReadMaxGrowth(
            DdManager *dd)
    void Cudd_SetMaxGrowth(
            DdManager *dd, double mg)
    unsigned int Cudd_ReadMinHit(
            DdManager *dd)
    void Cudd_SetMinHit(
            DdManager *dd, unsigned int hr)
    void Cudd_EnableGarbageCollection(
            DdManager *dd)
    void Cudd_DisableGarbageCollection(
            DdManager *dd)
    int Cudd_GarbageCollectionEnabled(
            DdManager * dd)
    unsigned int Cudd_ReadLooseUpTo(
            DdManager *dd)
    void Cudd_SetLooseUpTo(
            DdManager *dd, unsigned int lut)
    # quantification
    DdNode *Cudd_bddExistAbstract(
            DdManager *manager,
            DdNode *f,
            DdNode *cube)
    DdNode *Cudd_bddUnivAbstract(
            DdManager *manager,
            DdNode *f,
            DdNode *cube)
    DdNode *Cudd_bddAndAbstract(
            DdManager *manager,
            DdNode *f, DdNode *g,
            DdNode *cube)
    DdNode *Cudd_bddSwapVariables(
            DdManager *dd,
            DdNode *f, DdNode ** x, DdNode ** y,
            int n)

ctypedef DdNode *DdRef
cdef CUDD_UNIQUE_SLOTS = 2 ** 8
cdef CUDD_CACHE_SLOTS = 2 ** 18
cdef CUDD_REORDER_GROUP_SIFT = 14
cdef CUDD_OUT_OF_MEM = -1
cdef MAX_CACHE = <unsigned int> - 1

cdef extern from 'dddmp.h':
    ctypedef enum Dddmp_VarInfoType:
        pass
    ctypedef enum Dddmp_VarMatchType:
        pass
    int Dddmp_cuddBddStore(
            DdManager *ddMgr,
            char *ddname,
            DdNode *f,
            char ** varnames,
            int *auxids,
            int mode,
            Dddmp_VarInfoType varinfo,
            char *fname,
            FILE *fp)
    DdNode *Dddmp_cuddBddLoad(
            DdManager *ddMgr,
            Dddmp_VarMatchType varMatchMode,
            char ** varmatchnames,
            int *varmatchauxids,
            int *varcomposeids,
            int mode,
            char *fname,
            FILE *fp)

# 2**30 = 1 GiB (gibibyte, read ISO/IEC 80000)
DEFAULT_MEMORY = 1 * 2 ** 30

# -------------------- LDD -----------------------
# Copyright (c) 2009 Carnegie Mellon University.
# All rights reserved.
cdef extern from "ldd.h":
    ctypedef void * lincons_t
    ctypedef void * constant_t
    ctypedef void * linterm_t
    ctypedef void * qelim_context_t
    ctypedef DdNode LddNode
    ctypedef DdNode LddNodeset

    ctypedef theory theory_t

    ctypedef struct theory:
        constant_t (*create_int_cst)(int v);
        constant_t (*create_rat_cst)(long n, long d);
        constant_t (*create_double_cst)(double v);
        signed long int(*cst_get_si_num)(constant_t c);
        signed long int(*cst_get_si_den)(constant_t c);
        constant_t(*dup_cst)(constant_tc);
        constant_t(*negate_cst)(constant_t c);
        constant_t(*floor_cst)(constant_t c);
        constant_t(*ceil_cst)(constant_t c);
        int(*sgn_cst)(constant_t c);
        void(*destroy_cst)(constant_t c);
        constant_t(*add_cst)(constant_t c1, constant_t c2);
        constant_t(*mul_cst)(constant_t c1, constant_t c2);
        linterm_t(*create_linterm)(int * coeffs, size_t n);
        int(*term_size)(linterm_t t);
        int(*term_get_var)(linterm_t t, int i);
        constant_t(*term_get_coeff)(linterm_t t, int i);
        constant_t(*var_get_coeff)(linterm_t t, int x);
        linterm_t(*create_linterm_sparse_si)(int * var, int * coeff, size_t n);
        linterm_t(*create_linterm_sparse)(int * var, constant_t * coeff, size_t n);
        linterm_t(*dup_term)(linterm_t t);
        int(*term_equals)(linterm_t t1, linterm_t t2);
        int(*term_has_var)(linterm_t t, int var);
        int(*term_has_vars)(linterm_t t, int * vars);
        size_t(*num_of_vars)(theory_t * self);
        int(*terms_have_resolvent)(linterm_t t1, linterm_t t2, int x);
        linterm_t(*negate_term)(linterm_t t);
        void(*destroy_term)(linterm_t t);
        lincons_t(*create_cons)(linterm_t t, int s, constant_t k);
        int(*is_strict)(lincons_t l);
        void(*var_occurrences)(lincons_t l, int * occurs);
        linterm_t(*get_term)(lincons_t l);
        constant_t(*get_constant)(lincons_t l);
        lincons_t(*floor_cons)(lincons_t l);
        lincons_t(*ceil_cons)(lincons_t l);
        lincons_t(*negate_cons)(lincons_t l);
        int(*is_negative_cons)(lincons_t l);
        int(*is_stronger_cons)(lincons_t l1, lincons_t l2);
        lincons_t(*resolve_cons)(lincons_t l1, lincons_t l2, int x);
        void(*destroy_lincons)(lincons_t l);
        lincons_t(*dup_lincons)(lincons_t l);
        LddNode * (*to_ldd)(LddManager * m, lincons_t l);
        LddNode * (*subst)(LddManager * m, lincons_t l, int x, linterm_t t, constant_t c);
        LddNode * (*subst_pluse)(LddManager * m, lincons_t l, int x, linterm_t t, constant_t c);
        LddNode * (*subst_ninf)(LddManager * m, lincons_t l, int x);
        void(*var_bound)(lincons_t l, int x, linterm_t * dt, constant_t * dc);
        void(*print_lincons)(FILE * f, lincons_t l);
        int(*print_lincons_smtlibv1)(FILE * fp, lincons_t l, char ** vnames);
        # int(*dump_smtlibv1_prefix)(theory_t * self, FILE * fp, int * occrrences);
        void(*theory_debug_dump)(LddManager * tdd);
        qelim_context_t * (*qelim_init)(LddManager * m, int * vars);
        void(*qelim_push)(qelim_context_t * ctx, lincons_t l);
        lincons_t(*qelim_pop)(qelim_context_t * ctx);
        LddNode * (*qelim_solve)(qelim_context_t * ctx);
        void(*qelim_destroy_context)(qelim_context_t * ctx);

cdef extern from "lddInt.h":
    ctypedef struct LddManager:
        DdManager *cudd
        lincons_t *ddVars
        size_t varsSize
        bool be_bddlike
        theory_t *theory


cdef extern from "ldd.h":
    LddManager * Ldd_Init(DdManager *cudd, theory_t * t)
    void Ldd_Quit(LddManager *ldd)

    theory_t *Ldd_GetTheory(LddManager *ldd)
    LddNode * Ldd_FromCons(LddManager * m, lincons_t l);
    LddNode * Ldd_NewVar(LddManager * m, lincons_t l);
    LddNode * Ldd_NewVarAtTop(LddManager * m, lincons_t l);
    LddNode * Ldd_NewVarBefore(LddManager * m, LddNode * v, lincons_t l);
    LddNode * Ldd_NewVarAfter(LddManager * m, LddNode * v, lincons_t l);

    LddNode *Ldd_GetTrue(LddManager *m);
    LddNode *Ldd_GetFalse(LddManager *m);

    LddNode * Ldd_And(LddManager * m, LddNode * n1, LddNode * n2);
    LddNode * Ldd_Or(LddManager * m, LddNode * n1, LddNode * n2);
    LddNode * Ldd_Xor(LddManager * m, LddNode * n1, LddNode * n2);
    LddNode * Ldd_Ite(LddManager * m, LddNode * n1, LddNode * n2, LddNode * n3);

    LddNode * Ldd_ExistsAbstract(LddManager *, LddNode *, int);
    LddNode * Ldd_UnivAbstract(LddManager *, LddNode *, int);

    LddNode * Ldd_ExistsAbstractLW(LddManager *, LddNode *, int);
    LddNode * Ldd_ExistsAbstractFM(LddManager *, LddNode *, int);
    LddNode * Ldd_ExistsAbstractSFM(LddManager *, LddNode *, int);

    LddNode * Ldd_ExistAbstractPAT(LddManager *, LddNode *, int *);

    LddNode * Ldd_ResolveElim(LddManager *, LddNode *, linterm_t,
                              lincons_t, int);
    LddNode * Ldd_Resolve(LddManager *, LddNode *,
                          linterm_t, lincons_t, lincons_t, int);

    void Ldd_ManagerDebugDump(LddManager *);
    int Ldd_PathSize(LddManager *, LddNode *);

    void Ldd_SanityCheck(LddManager *);
    void Ldd_NodeSanityCheck(LddManager *, LddNode *);

    LddNode *Ldd_SatReduce(LddManager *, LddNode *, int);
    bool Ldd_IsSat(LddManager *, LddNode *);
    int Ldd_UnsatSize(LddManager *, LddNode *);
    theory_t *Ldd_SyntacticImplicationTheory(theory_t *t);
    void Ldd_VarOccurrences(LddManager *, LddNode *, int *);
    LddNode *Ldd_BddExistAbstract(LddManager *, LddNode *, LddNode *);
    LddNode *Ldd_TermsWithVars(LddManager *, int *);
    LddNode *Ldd_OverAbstract(LddManager *, LddNode *, int *);
    void Ldd_SupportVarOccurrences(LddManager *, LddNode *, int *);
    LddManager * Ldd_BddlikeManager(LddManager *);
    LddManager * Ldd_SetExistsAbstract(LddManager *,
                                       LddNode *(*)(LddManager *, LddNode *, int));
    LddNode * Ldd_MvExistAbstract(LddManager *, LddNode *, int *, size_t);
    LddNode * Ldd_BoxExtrapolate(LddManager *, LddNode *, LddNode *);
    LddNode * Ldd_BoxWiden(LddManager *, LddNode *, LddNode *);
    LddNode * Ldd_BoxWiden2(LddManager *, LddNode *, LddNode *);
    LddNode * Ldd_IntervalWiden(LddManager *, LddNode *, LddNode *);

    LddNode * Ldd_TermReplace(LddManager *, LddNode *, linterm_t, linterm_t, constant_t, constant_t, constant_t);
    LddNode * Ldd_TermCopy(LddManager *, LddNode *, linterm_t, linterm_t);
    LddNode * Ldd_TermMinmaxApprox(LddManager *, LddNode *);
    LddNode * Ldd_TermConstrain(LddManager *, LddNode *, linterm_t, linterm_t, constant_t);
    LddNodeset * Ldd_NodesetUnion(LddManager *, LddNodeset *, LddNodeset *);
    LddNodeset * Ldd_NodesetAdd(LddManager *, LddNodeset *, LddNode *);
    int Ldd_PrintMinterm(LddManager *, LddNode *);

    DdManager * Ldd_GetCudd(LddManager *);
    lincons_t Ldd_GetCons(LddManager *, LddNode *);

    LddNode * Ldd_SubstNinfForVar(LddManager *, LddNode *, int);
    LddNode * Ldd_SubstTermForVar(LddManager *, LddNode *, int, linterm_t, constant_t);
    LddNode * Ldd_SubstTermPlusForVar(LddManager *, LddNode *, int, linterm_t, constant_t);
    int Ldd_DumpSmtLibV1(LddManager *, LddNode *, char **, char *, FILE *);
    LddNode * Ldd_Cofactor(LddManager *, LddNode *, LddNode *);
    int Ldd_TermLeq(LddManager *, LddNode *, LddNode *);

cdef extern from "tvpi.h":
    theory_t *tvpi_create_theory(size_t vn);
    theory_t *tvpi_create_utvpiz_theory(size_t vn);
    theory_t *tvpi_create_tvpiz_theory(size_t vn);
    theory_t *tvpi_create_box_theory(size_t vn);
    theory_t *tvpi_create_boxz_theory(size_t vn);

    void tvpi_destroy_theory(theory_t *);

logger = logging.getLogger(__name__)

# -------------------- Constant ----------------
cdef constant_t Constant(
        ldd: LDD,
        k: _Constant):
    logger.debug(f'Creating constant {k}')
    cdef constant_t constant = NULL
    if type(k) is int:
        constant = ldd.ldd_theory.create_int_cst(k)
    elif type(k) is float:
        constant = ldd.ldd_theory.create_double_cst(k)
    else:
        constant = ldd.ldd_theory.create_rat_cst(k[0], k[1])
    if constant is NULL:
        raise RuntimeError('Failed to create constant')
    logger.debug('Constant created')
    return constant

    def __dealloc__(self):
        # Notice: constants are destroyed automatically on Ldd_Quit()
        pass

# -------------------- Linear Term ----------------
cdef linterm_t LinearTerm(
        ldd: LDD,
        t: _LinearTerm):
    logger.debug(f'Creating linear term {t}')
    cdef int n = len(t)
    cdef int *coeffs = <int *> PyMem_Malloc(n * sizeof(int))
    if coeffs is NULL:
        raise MemoryError('Failed to allocate memory for linear term')
    cdef int i
    for i in range(n):
        coeffs[i] = t[i]
    cdef linterm_t linterm = ldd.ldd_theory.create_linterm(coeffs, n)
    PyMem_Free(coeffs)
    if linterm is NULL:
        raise RuntimeError('Failed to create linear term')
    return linterm

# -------------------- Linear Constraint ----------------
cdef lincons_t LinearConstraint(ldd: LDD,
                                t: _LinearTerm,
                                s: bool,
                                k: _Constant):
    logger.debug(f'Creating linear constraint {t} {s} {k}')
    if len(t) > ldd.n_theory_vars + ldd.n_bool_vars:
        raise ValueError(
            'len(term) > n_theory_vars + n_bool_vars: '
            f'{len(t)} > {ldd.n_theory_vars} + {ldd.n_bool_vars}')
    cdef linterm_t linterm = LinearTerm(ldd, t)
    cdef constant_t constant = Constant(ldd, k)
    cdef lincons_t lincons = ldd.ldd_theory.create_cons(linterm, s, constant)
    if lincons is NULL:
        raise RuntimeError('Failed to create linear constraint')
    logger.debug('Linear constraint created')
    return lincons

# -------------------- LDD Manager ----------------
cpdef enum TheoryType:
    TVPI = 0
    TVPIZ = 1
    UTVPIZ = 2
    BOX = 3
    BOXZ = 4

cdef class LDD:
    """Wrapper of CUDD manager.

    Interface similar to `dd.cudd.BDD`.
    Variable names are strings.
    Attributes:

      - `vars`: `set` of T-variable names as `str`ings
    """

    cdef DdManager *cudd_manager
    cdef LddManager *ldd_manager
    cdef theory_t *ldd_theory
    cdef int n_theory_vars
    cdef int n_bool_vars
    # vars are strings representing theory-constraints
    cdef public object cons
    cdef public object _index_of_cons
    cdef public object _cons_with_index

    # bool_vars are strings representing theory-constraints
    cdef public object bool_vars
    cdef public object _index_of_bool_vars
    cdef public object _bool_vars_with_index

    def __cinit__(
            self,
            theory: TheoryType,
            n_theory_vars: int,
            n_bool_vars: int,
            memory_estimate: _NumberOfBytes | None = None,
            initial_cache_size: _Cardinality | None = None,
            *arg,
            **kw
    ) -> None:
        """Initialize LDD manager.

        @param theory:
            type of theory to use.
        @param nvars:
            number of theory variables.
        @param memory_estimate:
            maximum allowed memory, in bytes.
            If `None`, then use `DEFAULT_MEMORY`.
        @param initial_cache_size:
            initial number of slots in cache.
            If `None`, then use `CUDD_CACHE_SLOTS`.
        """
        self.cudd_manager = NULL  # prepare for `__dealloc__`, see cudd.BDD
        self.ldd_manager = NULL
        self.ldd_theory = NULL

        total_memory = _utils.total_memory()
        default_memory = DEFAULT_MEMORY
        if memory_estimate is None:
            memory_estimate = default_memory
        if memory_estimate >= total_memory:
            msg = (
                f'Error in `dd.ldd`: total physical memory is {total_memory} bytes, '
                f'but requested {memory_estimate} bytes. Please pass an amount of memory to '
                'the `LDD` constructor to avoid this error. For example, by instantiating '
                f'the `LDD` manager as `LDD({round(total_memory / 2)})`.')
            raise ValueError(msg)
        if initial_cache_size is None:
            initial_cache_size = CUDD_CACHE_SLOTS
        initial_subtable_size = CUDD_UNIQUE_SLOTS
        initial_n_vars_bdd = 0
        initial_n_vars_zdd = 0
        cudd_mgr = Cudd_Init(
            initial_n_vars_bdd,
            initial_n_vars_zdd,
            initial_subtable_size,
            initial_cache_size,
            memory_estimate)
        if cudd_mgr is NULL:
            raise RuntimeError('failed to initialize CUDD DdManager')

        # theory
        ldd_theory = NULL
        initial_n_vars = n_theory_vars + n_bool_vars

        if theory == TheoryType.TVPI:
            ldd_theory = tvpi_create_theory(initial_n_vars)
        elif theory == TheoryType.TVPIZ:
            ldd_theory = tvpi_create_tvpiz_theory(initial_n_vars)
        elif theory == TheoryType.UTVPIZ:
            ldd_theory = tvpi_create_utvpiz_theory(initial_n_vars)
        elif theory == TheoryType.BOX:
            ldd_theory = tvpi_create_box_theory(initial_n_vars)
        elif theory == TheoryType.BOXZ:
            ldd_theory = tvpi_create_boxz_theory(initial_n_vars)
        else:
            raise ValueError('invalid theory type')
        if ldd_theory is NULL:
            raise RuntimeError('Failed to initialize TVPI theory')
        self.ldd_theory = ldd_theory
        self.n_theory_vars = n_theory_vars
        self.n_bool_vars = n_bool_vars

        ldd_mgr = Ldd_Init(cudd_mgr, self.ldd_theory)
        if ldd_mgr is NULL:
            raise RuntimeError('failed to initialize LDD manager')
        self.cudd_manager = cudd_mgr
        self.ldd_manager = ldd_mgr

    def __init__(
            self,
            theory: TheoryType,
            n_theory_vars: int,
            n_bool_vars: int,
            memory_estimate: _NumberOfBytes | None = None,
            initial_cache_size: _Cardinality | None = None,
    ) -> None:
        self.configure(max_cache_hard=MAX_CACHE)
        self.cons: set[_VariableName] = set()
        self._index_of_cons: dict[_VariableName, int] = {}
        self._cons_with_index: dict[int, _VariableName] = {}

        self.bool_vars: set[_VariableName] = set()
        self._index_of_bool_vars: dict[_VariableName, int] = {}
        self._bool_vars_with_index: dict[int, _VariableName] = {}

    def __dealloc__(
            self
    ) -> None:
        logger.debug("LDD __dealloc__")
        null_err_msg = ('`{}` is `NULL`, which suggests that an exception was '
                        'raised inside the method `dd.ldd.LDD.__cinit__`.')

        if self.ldd_manager is NULL:
            raise RuntimeError(null_err_msg.format('self.ldd_manager'))
        Ldd_Quit(self.ldd_manager)

        if self.ldd_theory is NULL:
            raise RuntimeError(null_err_msg.format('self.ldd_theory'))
        tvpi_destroy_theory(self.ldd_theory)

        if self.cudd_manager is NULL:
            raise RuntimeError(null_err_msg.format('self.cudd_manager'))
        n = Cudd_CheckZeroRef(self.cudd_manager)
        if n != 0:
            logging.warning(f'Still {n} nodes referenced upon shutdown.')
        Cudd_Quit(self.cudd_manager)
        logger.debug("LDD __dealloc__ done")

    def __eq__(
            self: LDD,
            other: _ty.Optional[LDD]
    ) -> _Yes:
        """Return `True` if `other` has same manager."""
        if other is None:
            return False
        return self.ldd_manager == other.ldd_manager

    def __ne__(
            self: LDD,
            other: _ty.Optional[LDD]
    ) -> _Yes:
        return not (self == other)

    def __len__(
            self
    ) -> _Cardinality:
        """Number of nodes with nonzero references."""
        return Cudd_CheckZeroRef(self.cudd_manager)

    def __contains__(
            self,
            u: Function
    ) -> _Yes:
        if u.cudd_manager != self.cudd_manager:
            raise ValueError('undefined containment, because `u.cudd_manager != self.cudd_manager`')
        try:
            Cudd_NodeReadIndex(u.node)
            return True
        except:
            return False

    def __str__(
            self
    ) -> str:
        d = self.statistics()
        s = (
            'Linear decision diagram '
            '(LDD wrapper) with:\n'
            '\t {n} live nodes now\n'
            '\t {peak} live nodes at peak\n'
            '\t {n_cons} LDD constraints nodes\n'
            '\t {n_vars} LDD theory variables\n'
            '\t {mem:10.1f} bytes in use\n'
            '\t {reorder_time:10.1f} sec '
            'spent reordering\n'
            '\t {n_reorderings} reorderings\n'
        ).format(
            n=d['n_nodes'],
            peak=d['peak_live_nodes'],
            n_cons=d['n_cons'],
            n_vars=d['n_vars'],
            reorder_time=d['reordering_time'],
            n_reorderings=d['n_reorderings'],
            mem=d['mem'])
        return s

    def statistics(
            self,
            exact_node_count: _Yes = False
    ) -> dict[str, _ty.Any]:
        """Return `dict` with LDD node counts and times.

        If `exact_node_count` is `True`, then the
        list of dead nodes is cleared.

        Keys with meaning:

          - `n_vars`: number of theory variables
          - `n_cons`: number of constraints
          - `n_nodes`: number of live nodes
          - `peak_nodes`: max number of all nodes
          - `peak_live_nodes`: max number of live nodes

          - `reordering_time`: sec spent reordering
          - `n_reorderings`: number of reorderings

          - `mem`: bytes in use
          - `unique_size`: total number of
            buckets in unique table
          - `unique_used_fraction`: buckets that
            contain >= 1 node
          - `expected_unique_used_fraction`:
            if properly working

          - `cache_size`: number of slots in cache
          - `cache_used_fraction`: slots with data
          - `cache_lookups`: total number of lookups
          - `cache_hits`: total number of cache hits
          - `cache_insertions`
          - `cache_collisions`
          - `cache_deletions`
        """
        cdef DdManager *mgr
        mgr = self.cudd_manager
        n_cons = Cudd_ReadSize(mgr)
        # call cdef theory.num_of_vars(). Remember self.ldd_theory is a theory_t*.
        n_vars = 0  # self.ldd_theory.num_of_vars()
        # nodes
        if exact_node_count:
            n_nodes = Cudd_ReadNodeCount(mgr)
        else:
            n_nodes = mgr.keys - mgr.dead
        peak_nodes = Cudd_ReadPeakNodeCount(mgr)
        peak_live_nodes = Cudd_ReadPeakLiveNodeCount(mgr)
        # reordering
        t = Cudd_ReadReorderingTime(mgr)
        reordering_time = t / 1000.0
        n_reorderings = Cudd_ReadReorderings(mgr)
        # memory
        m = Cudd_ReadMemoryInUse(mgr)
        mem = float(m)
        # unique table
        unique_size = Cudd_ReadSlots(mgr)
        unique_used_fraction = Cudd_ReadUsedSlots(mgr)
        expected_unique_fraction = (
            Cudd_ExpectedUsedSlots(mgr))
        # cache
        cache_size = Cudd_ReadCacheSlots(mgr)
        cache_used_fraction = Cudd_ReadCacheUsedSlots(mgr)
        cache_lookups = Cudd_ReadCacheLookUps(mgr)
        cache_hits = Cudd_ReadCacheHits(mgr)
        cache_insertions = mgr.cacheinserts
        cache_collisions = mgr.cachecollisions
        cache_deletions = mgr.cachedeletions
        d = dict(
            n_cons=n_cons,
            n_vars=n_vars,
            n_nodes=n_nodes,
            peak_nodes=peak_nodes,
            peak_live_nodes=peak_live_nodes,
            reordering_time=reordering_time,
            n_reorderings=n_reorderings,
            mem=mem,
            unique_size=unique_size,
            unique_used_fraction=unique_used_fraction,
            expected_unique_used_fraction=
            expected_unique_fraction,
            cache_size=cache_size,
            cache_used_fraction=cache_used_fraction,
            cache_lookups=cache_lookups,
            cache_hits=cache_hits,
            cache_insertions=cache_insertions,
            cache_collisions=cache_collisions,
            cache_deletions=cache_deletions)
        return d

    def configure(self, **kw) -> dict[str, _ty.Any]:
        """see cudd.BDD.configure
        """
        cdef int method
        cdef DdManager *mgr
        mgr = self.cudd_manager
        # read
        reordering = Cudd_ReorderingStatus(mgr, <Cudd_ReorderingType *> &method)
        garbage_collection = (Cudd_GarbageCollectionEnabled(mgr))
        max_memory = Cudd_ReadMaxMemory(mgr)
        loose_up_to = Cudd_ReadLooseUpTo(mgr)
        max_cache_soft = Cudd_ReadMaxCache(mgr)
        max_cache_hard = Cudd_ReadMaxCacheHard(mgr)
        min_hit = Cudd_ReadMinHit(mgr)
        max_growth = Cudd_ReadMaxGrowth(mgr)
        max_swaps = Cudd_ReadSiftMaxSwap(mgr)
        max_vars = Cudd_ReadSiftMaxVar(mgr)
        d = dict(
            reordering=reordering == 1,
            garbage_collection=garbage_collection == 1,
            max_memory=max_memory,
            loose_up_to=loose_up_to,
            max_cache_soft=max_cache_soft,
            max_cache_hard=max_cache_hard,
            min_hit=min_hit,
            max_growth=max_growth,
            max_swaps=max_swaps,
            max_vars=max_vars)
        # set
        for k, v in kw.items():
            if k == 'reordering':
                if v:
                    Cudd_AutodynEnable(mgr, CUDD_REORDER_GROUP_SIFT)
                else:
                    Cudd_AutodynDisable(mgr)
            elif k == 'garbage_collection':
                if v:
                    Cudd_EnableGarbageCollection(mgr)
                else:
                    Cudd_DisableGarbageCollection(mgr)
            elif k == 'max_memory':
                Cudd_SetMaxMemory(mgr, v)
            elif k == 'loose_up_to':
                Cudd_SetLooseUpTo(mgr, v)
            elif k == 'max_cache_hard':
                Cudd_SetMaxCacheHard(mgr, v)
            elif k == 'min_hit':
                Cudd_SetMinHit(mgr, v)
            elif k == 'max_growth':
                Cudd_SetMaxGrowth(mgr, v)
            elif k == 'max_swaps':
                Cudd_SetSiftMaxSwap(mgr, v)
            elif k == 'max_vars':
                Cudd_SetSiftMaxVar(mgr, v)
            elif k == 'max_cache_soft':
                logger.warning(
                    '"max_cache_soft" not settable.')
            else:
                raise ValueError(
                    f'Unknown parameter "{k}"')
        return d

    cpdef tuple succ(
            self,
            u: Function):
        """Return `(level, low, high)` for `u`."""
        self._assert_this_manager(u)
        i = u.level
        v = u.low
        w = u.high
        return i, v, w

    cpdef incref(
            self,
            u: Function):
        """Increment the reference count of `u`.

        Raise `RuntimeError` if `u._ref <= 0`.
        For more details about avoiding this read the docstring of the class `Function`.

        The reference count of the BDD node in CUDD that `u` points to is incremented.

        Also, the attribute `u._ref` is incremented.

        Calling this method is unnecessary, because reference counting is automated.
        """
        if u.node is NULL:
            raise RuntimeError('`u.node` is `NULL` pointer.')
        if u._ref <= 0:
            _utils._raise_runtimerror_about_ref_count(
                u._ref, 'method `dd.cudd.BDD.incref`',
                '`dd.cudd.Function`')
        assert u._ref > 0, u._ref
        u._ref += 1
        self._incref(u.node)

    cpdef decref(
            self,
            u: Function,
            recursive: _Yes = False,
            _direct: _Yes = False):
        """Decrement the reference count of `u`.

        Raise `RuntimeError` if `u._ref <= 0` or `u.node is NULL`.
        For more details about avoiding this read the docstring of the class `Function`.

        The reference count of the BDD node in CUDD that `u` points to is decremented.

        Also, the attribute `u._ref` is decremented.
        If after this decrement, `u._ref == 0`, then the pointer `u.node` is set to `NULL`.

        Calling this method is unnecessary, because reference counting is automated.

        If early dereferencing of the node is desired in order to allow garbage collection,
        then write `del u`, instead of calling this method.

        @param recursive:
            if `True`, then call
            `Cudd_IterDerefBdd`,
            else call `Cudd_Deref`
        @param _direct:
            use this parameter only after
            reading the source code of the
            Cython file `dd/cudd.pyx`.
            When `_direct == True`, some of the above
            description does not apply.
        """
        if u.node is NULL:
            raise RuntimeError('`u.node` is `NULL` pointer.')
        # bypass checks and leave `u._ref` unchanged,
        # directly call `_decref`
        if _direct:
            self._decref(u.node, recursive)
            return
        if u._ref <= 0:
            _utils._raise_runtimerror_about_ref_count(
                u._ref, 'method `dd.cudd.BDD.decref`',
                '`dd.cudd.Function`')
        assert u._ref > 0, u._ref
        u._ref -= 1
        self._decref(u.node, recursive)
        if u._ref == 0:
            u.node = NULL

    cdef _incref(
            self,
            u: DdRef):
        Cudd_Ref(u)

    cdef _decref(
            self,
            u: DdRef,
            recursive: _Yes = False):
        # There is little point in checking here the reference count of `u`, because
        # doing that relies on the assumption that `u` still corresponds to a node,
        # which implies that the reference count is positive.
        #
        # This point should not be reachable after `u` reaches zero reference count.
        #
        # Moreover, if the memory has been deallocated, then in principle the attribute `ref`
        # can have any value, so an assertion here would not be ensuring correctness.
        if recursive:
            Cudd_IterDerefBdd(self.cudd_manager, u)
        else:
            Cudd_Deref(u)

    @property
    def vars(
            self
    ) -> _VariableNames:
        return self.cons

    cpdef Function bool_var(
            self,
            name: str):
        """Return node for boolean variable named `name`.
        Boolean variables are encoded as theory constraints (x_i <= 0), where
        x_i is a fresh theory variable.
        """
        logger.debug(f'Creating boolean var: {name}')
        if name not in self.bool_vars:
            idx = self.n_theory_vars + len(self.bool_vars)
            self.bool_vars.add(name)
            self._index_of_bool_vars[name] = idx
        idx = self._index_of_bool_vars[name]
        coeff = [0] * idx
        coeff.append(1)
        cons = (tuple(coeff), False, 0)
        return self.constraint(cons, name)

    cpdef Function constraint(
            self,
            cons: _LinearConstraint,
            name: str = None,
    ):
        logger.debug(f'Creating constraint: {cons}')
        term, strict, constant = cons
        cdef lincons_t cons_ptr = LinearConstraint(self, term, strict, constant)
        cdef LddNode *cldd = Ldd_FromCons(self.ldd_manager, cons_ptr)
        idx = Cudd_NodeReadIndex(cldd)
        cons_ptr = self.ldd_manager.ddVars[idx]
        varname = name or self.get_cons_str(cons_ptr).decode('utf-8')
        if varname not in self.cons:
            self.cons.add(varname)
            self._index_of_cons[varname] = idx
            self._cons_with_index[idx] = varname
            logger.debug('Constraint created with index: ' + str(idx))
        else:
            idx = self._index_of_cons[varname]
            logger.debug('Constraint already exists with index: ' + str(idx))
        return wrap(self, cldd, is_leaf=True)

    cpdef Function var(
            self,
            var: _VariableName):
        """Return node for variable named `var`."""
        if var not in self._index_of_cons and var not in self._index_of_bool_vars:
            raise ValueError(
                f'undeclared variable "{var}", '
                'the declared variables are:\n'
                f'{self._index_of_cons | self._index_of_bool_vars}')
        j = self._index_of_cons[var] if var in self._index_of_cons else self._index_of_bool_vars[var]
        r = Cudd_bddIthVar(self.cudd_manager, j)
        if r == NULL:
            raise RuntimeError(
                f'failed to create LDD for variable "{var}"')
        return wrap(self, r)

    def var_at_level(
            self,
            level: _Level
    ) -> _VariableName:
        """Return name of variable at `level`.

        Raise `ValueError` if `level` is not
        the level of any variable declared in
        `self.vars`.
        """
        j = Cudd_ReadInvPerm(self.cudd_manager, level)
        if (j == -1 or j == CUDD_CONST_INDEX or
                j not in self._cons_with_index):
            raise ValueError(_tw.dedent(f'''
                    No declared variable has level: {level}.
                    {_utils.var_counts(self)}
                    '''))
        var = self._cons_with_index[j]
        return var

    def level_of_var(
            self,
            var:
            _VariableName
    ) -> _Level:
        """Return level of variable named `var`.

        Raise `ValueError` if `var` is not
        a variable in `self.vars`.
        """
        if var not in self._index_of_cons:
            raise ValueError(
                f'undeclared variable "{var}", '
                'the declared variables are:\n'
                f'{self._index_of_cons}')
        j = self._index_of_cons[var]
        level = Cudd_ReadPerm(self.cudd_manager, j)
        if level == -1:
            raise AssertionError(
                f'index {j} out of bounds for variable "{var}"')
        return level

    @property
    def var_levels(
            self
    ) -> _VariableLevels:
        return {
            var: self.level_of_var(var)
            for var in self.cons}

    def _number_of_cudd_vars(
            self
    ) -> _Cardinality:
        """Return number of CUDD indices.

        Can be `> len(self.vars)`.
        """
        n_cudd_vars = Cudd_ReadSize(self.cudd_manager)
        if 0 <= n_cudd_vars <= CUDD_CONST_INDEX:
            return n_cudd_vars
        raise RuntimeError(_tw.dedent(f'''
            Unexpected value: {n_cudd_vars}
            returned from `Cudd_ReadSize()`
            (expected <= {CUDD_CONST_INDEX} =
             CUDD_CONST_INDEX)
            '''))

    def reorder(
            self,
            var_order:
            _VariableLevels |
            None = None
    ) -> None:
        raise NotImplementedError("Reordering is not implemented for LDD")

    cdef set support(
            self,
            u: Function):
        """Return constraints that `u` depends on.

        @return:
            set of constraints
        @rtype:
            `set[lincons_t]`
        """
        raise NotImplementedError()

    def group(
            self,
            vrs: _abc.Mapping[_LinearConstraint, _Nat]
    ) -> None:
        raise NotImplementedError("Grouping is not implemented for LDD")

    def copy(
            self,
            u: Function,
            other: LDD
    ) -> Function:
        """Transfer LDD with root `u` to `other`."""
        raise NotImplementedError("Copying is not yet implemented for LDD")

    cpdef Function let(
            self,
            definitions: _Renaming | _Assignment | dict[_LinearConstraint, Function],
            u: Function):
        raise NotImplementedError()

    cdef _assert_this_manager(
            self,
            u: Function):
        if u.ldd_manager != self.ldd_manager:
            raise ValueError(
                '`u.ldd_manager != self.ldd_manager`')

    cpdef Function ite(
            self,
            g: Function,
            u: Function,
            v: Function):
        """Ternary conditional.

        In other words, the root of the BDD that represents the expression:
        ```tla
        IF g THEN u ELSE v
        ```
        """
        self._assert_this_manager(g)
        self._assert_this_manager(u)
        self._assert_this_manager(v)
        r: DdRef
        r = Ldd_Ite(
            self.ldd_manager,
            g.node, u.node, v.node)
        return wrap(self, r)

    def count(
            self,
            u: Function,
            nvars: _Cardinality | None = None
    ) -> _Cardinality:
        """Return number of models of node `u`.

        @param nvars:
            regard `u` as an operator that depends on `nvars`-many variables.

            If omitted, then assume those variables in `support(u)`.
        """
        raise NotImplementedError()

    def pick(
            self,
            u: Function,
            care_vars: _abc.Set[_VariableName] | None = None
    ) -> _Assignment:
        """Return a single assignment.

        @return:
            assignment of values to variables
        """
        raise NotImplementedError()

    def _sat_iter(
            self,
            u: Function,
            cube: _Assignment,
            value: python_bool,
            support
    ) -> _abc.Iterable[_Assignment]:
        """Recurse to enumerate models."""
        raise NotImplementedError()

    cpdef Function apply(
            self,
            op: _dd_abc.OperatorSymbol,
            u: Function,
            v: _ty.Optional[Function] = None,
            w: _ty.Optional[Function] = None):
        """Return the result of applying `op`."""
        _utils.assert_operator_arity(op, v, w, 'bdd')
        self._assert_this_manager(u)
        if v is not None:
            self._assert_this_manager(v)
        if w is not None:
            self._assert_this_manager(w)
        r: DdRef
        cdef DdManager *cudd_mgr
        cudd_mgr = u.cudd_manager
        cdef LddManager *ldd_mgr
        ldd_mgr = u.ldd_manager
        # unary
        r = NULL
        if op in ('~', 'not', '!'):
            r = Cudd_Not(u.node)
        # binary
        elif op in ('and', '/\\', '&', '&&'):
            r = Ldd_And(ldd_mgr, u.node, v.node)
        elif op in ('or', r'\/', '|', '||'):
            r = Ldd_Or(ldd_mgr, u.node, v.node)
        elif op in ('#', 'xor', '^'):
            r = Ldd_Xor(ldd_mgr, u.node, v.node)
        elif op in ('=>', '->', 'implies'):
            r = Ldd_Ite(
                ldd_mgr, u.node, v.node,
                Ldd_GetTrue(ldd_mgr))
        elif op in ('<=>', '<->', 'equiv'):
            r = Cudd_Not(Ldd_Xor(ldd_mgr, u.node, v.node))
        elif op in ('diff', '-'):
            r = Ldd_Ite(
                ldd_mgr, u.node, Cudd_Not(v.node),
                Ldd_GetFalse(ldd_mgr))
        elif op in (r'\A', 'forall'):
            raise NotImplementedError(
                'universal quantification not implemented yet in this API')
        elif op in (r'\E', 'exists'):
            raise NotImplementedError(
                'existential quantification not implemented yet in this API')
        # ternary
        elif op == 'ite':
            r = Ldd_Ite(ldd_mgr, u.node, v.node, w.node)
        else:
            raise ValueError(f'unknown operator: "{op}"')
        if r is NULL:
            config = self.configure()
            raise RuntimeError((
                'LDD appears to have run out of memory.\n'
                'Current settings for  upper bounds are:\n'
                '    max memory = {max_memory} bytes\n'
                '    max cache = {max_cache} entries'
            ).format(
                max_memory=config['max_memory'],
                max_cache=config['max_cache_hard']))
        return wrap(self, r)

    cpdef Function cube(
            self,
            dvars: _abc.Collection[_LinearConstraint]):
        raise NotImplementedError()

    cpdef dict _cube_to_dict(
            self,
            f: Function):
        """Collect indices of support variables."""
        raise NotImplementedError()

    cpdef Function quantify(
            self,
            u: Function,
            qvars: _abc.Iterable[_VariableName],
            forall: _Yes = False):
        """Abstract variables `qvars` from node `u`."""
        raise NotImplementedError()

    cpdef Function forall(
            self,
            variables: _abc.Iterable[_VariableName],
            u: Function):
        """Quantify `variables` in `u` universally.

        Wraps method `quantify` to be more readable.
        """
        return self.quantify(u, variables, forall=True)

    cpdef Function exist(
            self,
            variables: _abc.Iterable[_VariableName],
            u: Function):
        """Quantify `variables` in `u` existentially.

        Wraps method `quantify` to be more readable.
        """
        return self.quantify(u, variables, forall=False)

    cpdef assert_consistent(self):
        """Raise `AssertionError` if not consistent."""
        if Cudd_DebugCheck(self.cudd_manager) != 0:
            raise AssertionError('`Cudd_DebugCheck` errored')

    def add_expr(
            self,
            expr: _Formula
    ) -> Function:
        """Return node for expression `e`."""
        raise NotImplementedError()

    cpdef str to_expr(
            self,
            u: Function):
        """Return a Boolean expression for node `u`."""
        self._assert_this_manager(u)
        cache = dict()
        return self._to_expr(u.node, cache)

    cdef str _to_expr(
            self,
            u: DdRef,
            cache: dict[int, str]):
        if u == Ldd_GetTrue(self.ldd_manager):
            return 'FALSE'
        if u == Ldd_GetFalse(self.ldd_manager):
            return 'TRUE'
        index = Cudd_NodeReadIndex(u)
        if index in cache:
            return cache[index]
        v = Cudd_E(u)
        w = Cudd_T(u)
        p = self._to_expr(v, cache)
        q = self._to_expr(w, cache)
        r = Cudd_Regular(u)
        cons_str = self._cons_with_index[index]
        # pure var ?
        if p == 'FALSE' and q == 'TRUE':
            expr = cons_str
        else:
            expr = f'ite({cons_str}, {q}, {p})'
        # complemented ?
        if Cudd_IsComplement(u):
            expr = f'(~ {expr})'
        cache[index] = expr
        return expr

    cdef char * get_cons_str(
            self,
            lincons_t cons):
        """Return string representation of constraint.
        """
        # Ldd offers only theory.print_lincons(file, cons).
        # Hence, we need to write to a file and read from it.
        cdef FILE *f = fopen('tmp.txt', 'w+')
        if f is NULL:
            raise MemoryError('Failed to open temporary file')
        cdef char *buf
        try:
            self.ldd_theory.print_lincons(f, cons)
            fseek(f, 0, SEEK_SET)
            size = 100
            buf = <char *> PyMem_Malloc(size * sizeof(char))
            if buf is NULL:
                raise MemoryError('Failed to allocate memory for constraint string')
            size_read = fread(buf, 1, size, f)

            buf[size_read] = 0
            # buf[size] = 0
            return buf
        finally:
            fclose(f)

    def dump(
            self,
            filename: str,
            roots: dict[str, Function] | list[Function],
            filetype: _BDDFileType | None = None
    ) -> None:
        """Write LDDs to `filename`.

        The file type is inferred from the extension (case-insensitive),
        unless a `filetype` is explicitly given.

        `filetype` can have the values:

        - `'pdf'` for PDF
        - `'png'` for PNG
        - `'svg'` for SVG
        - `'json'` for JSON
        - `'dddmp'` for DDDMP (of CUDD)

        If `filetype is None`, then `filename` must have an extension that matches
        one of the file types listed above.

        Dump nodes reachable from `roots`.

        Dumping a JSON file requires that `roots` be nonempty.

        Dumping a DDDMP file requires that `roots` contain a single node.

        @param roots:
            For JSON: a mapping from names to nodes.
        """

        if filetype is None:
            name = filename.lower()
            if name.endswith('.pdf'):
                filetype = 'pdf'
            elif name.endswith('.png'):
                filetype = 'png'
            elif name.endswith('.svg'):
                filetype = 'svg'
            elif name.endswith('.dot'):
                filetype = 'dot'
            elif name.endswith('.p'):
                raise ValueError(
                    'pickling unsupported '
                    'by this class, use JSON')
            elif name.endswith('.json'):
                filetype = 'json'
            elif name.endswith('.dddmp'):
                filetype = 'dddmp'
            else:
                raise ValueError(
                    'cannot infer file type '
                    'from extension of file '
                    f'name "{filename}"')
        if filetype == 'dddmp':
            raise NotImplementedError(
                'DDDMP dump is not yet implemented for LDD')
        elif filetype == 'json':
            if roots is None:
                raise ValueError(roots)
            _copy.dump_json(roots, filename)
            return
        elif (filetype != 'pickle' and
              filetype not in _utils.DOT_FILE_TYPES):
            raise ValueError(filetype)
        bdd = autoref.BDD()
        _copy.copy_vars(self, bdd)
        # preserve levels
        if roots is None:
            root_nodes = None
        else:
            cache = dict()
            def mapper(u):
                return _copy.copy_bdd(u, bdd, cache)
            root_nodes = _utils._map_container(mapper, roots)
        bdd.dump(filename, root_nodes, filetype=filetype)

    cpdef load(
            self,
            filename: str):
        """Return `Function` loaded from `filename`.

        @param filename:
            name of file from
            where the BDD is loaded
        @return:
            roots of loaded BDDs
        @rtype:
            depends on the contents of the file:
            | `dict[str, Function]`
            | `list[Function]`
        """
        raise NotImplementedError()

    @property
    def false(
            self
    ) -> Function:
        """Boolean value false."""
        return wrap(self, Ldd_GetFalse(self.ldd_manager))

    @property
    def true(
            self
    ) -> Function:
        """Boolean value true."""
        return wrap(self, Ldd_GetTrue(self.ldd_manager))

cpdef Function restrict(
        u: Function,
        care_set: Function):
    """Restrict `u` to `care_set`.

    The operator "restrict" is defined in
    1990 Coudert ICCAD.
    """
    raise NotImplementedError()

cpdef Function and_exists(
        u: Function,
        v: Function,
        qvars: _abc.Iterable[_VariableName]):
    r"""Return `\E qvars:  u /\ v`."""
    raise NotImplementedError()

cpdef Function or_forall(
        u: Function,
        v: Function,
        qvars: _abc.Iterable[_VariableName]):
    r"""Return `\A qvars:  u \/ v`."""
    raise NotImplementedError()

def copy_vars(
        source:
        LDD,
        target:
        LDD
) -> None:
    """Copy variables, preserving CUDD indices."""
    raise NotImplementedError()

cpdef int count_nodes(
        functions: list[Function]):
    """Return total nodes used by `functions`.

    Sharing is taken into account.
    """
    cdef DdRef *x
    f: Function
    n = len(functions)
    x = <DdRef *> PyMem_Malloc(n * sizeof(DdRef))
    for i, f in enumerate(functions):
        x[i] = f.node
    try:
        k = Cudd_SharingSize(x, n)
    finally:
        PyMem_Free(x)
    return k

cpdef dict count_nodes_per_level(
        bdd: LDD):
    """Return mapping of each var to a node count."""
    raise NotImplementedError()

cdef _dict_to_cube_array(
        d: _Assignment,
        int *x,
        index_of_var: _Assignment | set[_VariableName]):
    """Assign array of literals `x` from assignment `d`.

    @param x:
        array of literals
        0: negated, 1: positive, 2: don't care
        read `Cudd_FirstCube`
    @param index_of_var:
        `dict` from variables to `bool`
        or `set` of variable names.
    """
    raise NotImplementedError()

cdef dict _cube_array_to_dict(
        int *x,
        index_of_var:
        dict):
    """Return assignment from array of literals `x`.

    @param x:
        read `_dict_to_cube_array`
    """
    raise NotImplementedError()

cdef Function wrap(
        ldd: LDD,
        node: DdRef,
        is_leaf: bool = False):
    """Return a `Function` that wraps `node`."""
    f = Function()
    f.init(node, is_leaf, ldd)
    return f

cdef class Function:
    r"""see dd.cudd.Function"""

    __weakref__: object
    cdef public LDD ldd
    cdef DdManager *cudd_manager
    cdef LddManager *ldd_manager
    node: DdRef
    cdef public int _ref

    cdef init(
            self,
            node: DdRef,
            is_leaf: bool,
            ldd: LDD):
        if node is NULL:
            raise ValueError(
                '`DdNode *node` is `NULL` pointer.')
        logger.debug("Init Function")
        self.ldd = ldd
        self.cudd_manager = ldd.cudd_manager
        self.ldd_manager = ldd.ldd_manager
        self.node = node
        self._ref = 1  # lower bound on reference count
        #
        # Assumed invariant:
        # this instance participates in computation only as long as `self._ref > 0`.
        # The user is responsible for implementing this invariant.
        if not is_leaf:  # Leaf nodes are already referenced on creation (probably a bug of the C API)
            Cudd_Ref(node)
        logger.debug("Init Function done")

    def __hash__(
            self
    ) -> int:
        return int(self)

    @property
    def bdd(
            self
    ) -> LDD:
        """Return BDD that wraps `self.node`."""
        return self.ldd

    @property
    def _index(
            self
    ) -> _Nat:
        """Index of `self.node`."""
        return Cudd_NodeReadIndex(self.node)

    @property
    def var(
            self
    ) -> (_VariableName | None):
        """Variable at level where this node is.

        If node is constant, return `None`.
        """
        if Cudd_IsConstant(self.node):
            return None
        return self.ldd._cons_with_index[self._index]

    @property
    def level(
            self
    ) -> _Level:
        """Level where this node currently is."""
        i = self._index
        return Cudd_ReadPerm(self.cudd_manager, i)

    @property
    def ref(
            self
    ) -> _Cardinality:
        u: DdRef
        u = Cudd_Regular(self.node)
        return u.ref

    @property
    def low(
            self
    ) -> '''
                Function |
                None
                ''':
        """Return "else" node as `Function`."""
        u: DdRef
        if Cudd_IsConstant(self.node):
            return None
        u = Cudd_E(self.node)
        return wrap(self.ldd, u)

    @property
    def high(
            self
    ) -> '''
                Function |
                None
                ''':
        """Return "then" node as `Function`."""
        u: DdRef
        if Cudd_IsConstant(self.node):
            return None
        u = Cudd_T(self.node)
        return wrap(self.ldd, u)

    @property
    def negated(
            self
    ) -> _Yes:
        """`True` if this is a complemented edge.

        Returns `True` if `self` is
        a complemented edge.
        """
        return Cudd_IsComplement(self.node)

    @property
    def support(
            self:
            LDD
    ) -> set[_VariableName]:
        """Return `set` of variables in support."""
        return self.ldd.support(self)

    def __dealloc__(
            self
    ) -> None:
        logger.debug("Function __dealloc__")
        # when changing this method,
        # update also the function
        # `_test_call_dealloc` below
        if self._ref < 0:
            raise AssertionError(
                "The lower bound `_ref` "
                "on the node's "
                'reference count has '
                f'value {self._ref}, '
                'which is unexpected and '
                'should never happen. '
                'Was the value of `_ref` '
                'changed from outside '
                'this instance?')
        assert self._ref >= 0, self._ref
        if self._ref == 0:
            return
        if self.node is NULL:
            raise AssertionError(
                'The attribute `node` is '
                'a `NULL` pointer. '
                'This is unexpected and '
                'should never happen. '
                'Was the value of `_ref` '
                'changed from outside '
                'this instance?')
        # anticipate multiple calls to `__dealloc__`
        self._ref -= 1
        logger.debug("Cudd recursive deref")
        # deref
        Cudd_IterDerefBdd(
            self.cudd_manager, self.node)
        # avoid future access
        # to deallocated memory
        self.node = NULL
        logger.debug("Function __dealloc__ done")

    def __int__(
            self
    ) -> int:
        """Inverse of `BDD._add_int()`."""
        return _ddref_to_int(self.node)

    def __repr__(
            self
    ) -> str:
        u: DdRef
        u = Cudd_Regular(self.node)
        return (
            f'<dd.cudd.Function at {hex(id(self))}, '
            'wrapping a DdNode with '
            f'var index: {u.index}, '
            f'ref count: {u.ref}, '
            f'int repr: {int(self)}>')

    def __str__(
            self
    ) -> str:
        return f'@{int(self)}'

    def __len__(
            self
    ) -> _Cardinality:
        return Cudd_DagSize(self.node)

    @property
    def dag_size(
            self
    ) -> _Cardinality:
        """Return number of BDD nodes.

        This is the number of BDD nodes that
        are reachable from this BDD reference,
        i.e., with `self` as root.
        """
        return len(self)

    def __eq__(
            self: Function,
            other: _ty.Optional[Function]
    ) -> _Yes:
        if other is None:
            return False
        # guard against mixing managers
        self._assert_same_manager(other)
        return self.node == other.node

    def __ne__(
            self:
            Function,
            other:
            _ty.Optional[Function]
    ) -> _Yes:
        return not (self == other)

    def __le__(
            self: Function,
            other: Function
    ) -> _Yes:
        self._assert_same_manager(other)
        return (other | ~ self) == self.ldd.true

    def __lt__(
            self: Function,
            other: Function
    ) -> _Yes:
        self._assert_same_manager(other)
        return (self.node != other.node and
                (other | ~ self) == self.ldd.true)

    def __ge__(
            self: Function,
            other: Function
    ) -> _Yes:
        return other <= self

    def __gt__(
            self:
            Function,
            other:
            Function
    ) -> _Yes:
        return other < self

    def __invert__(
            self
    ) -> Function:
        r: DdRef
        r = Cudd_Not(self.node)
        return wrap(self.ldd, r)

    def __and__(
            self: Function,
            other: Function
    ) -> Function:
        self._assert_same_manager(other)
        r = Ldd_And(
            self.ldd_manager, self.node, other.node)
        return wrap(self.ldd, r)

    cdef _assert_same_manager(
            self,
            other: Function
    ):
        if self.ldd_manager != other.ldd_manager:
            raise ValueError(
                '`self.ldd_manager != other.ldd_manager`')

    def __or__(
            self: Function,
            other: Function
    ) -> Function:
        self._assert_same_manager(other)
        r = Ldd_Or(
            self.ldd_manager, self.node, other.node)
        return wrap(self.ldd, r)

    def implies(
            self: Function,
            other: Function
    ) -> Function:
        self._assert_same_manager(other)
        r = Ldd_Ite(
            self.ldd_manager, self.node,
            other.node, Ldd_GetTrue(self.ldd_manager))
        return wrap(self.ldd, r)

    def equiv(
            self:
            Function,
            other:
            Function
    ) -> Function:
        self._assert_same_manager(other)
        r = Ldd_Ite(
            self.ldd_manager, self.node,
            other.node, Cudd_Not(other.node))
        return wrap(self.ldd, r)

    def let(
            self:
            Function,
            **definitions: python_bool | Function
    ) -> Function:
        return self.ldd.let(definitions, self)

    def exist(
            self:
            Function,
            *variables: _VariableName
    ) -> Function:
        return self.ldd.exist(variables, self)

    def forall(
            self:
            Function,
            *variables: _VariableName
    ) -> Function:
        return self.ldd.forall(variables, self)

    def pick(
            self: Function,
            care_vars: _abc.Set[_VariableName] | None = None
    ) -> _Assignment:
        return self.ldd.pick(self, care_vars)

    def count(
            self: Function,
            nvars: _Cardinality | None = None
    ) -> _Cardinality:
        return self.ldd.count(self, nvars)

cdef _ddref_to_int(
        node: DdRef):
    """Convert node pointer to numeric index.

    Inverse of `_int_to_ddref()`.
    """
    if sizeof(stdint.uintptr_t) != sizeof(DdRef):
        raise AssertionError('mismatch of sizes')
    index = <stdint.uintptr_t> node
    # 0, 1 used to represent TRUE and FALSE
    # in syntax of expressions
    if 0 <= index:
        index += 2
    if index in (0, 1):
        raise AssertionError(index)
    return index

cdef DdRef _int_to_ddref(
        index: int):
    """Convert numeric index to node pointer.

    Inverse of `_ddref_to_int()`.
    """
    if index in (0, 1):
        raise ValueError(index)
    if 2 <= index:
        index -= 2
    u: DdRef = <DdRef> <stdint.uintptr_t> index
    return u

"""Tests and test wrappers for C functions."""

cpdef _test_incref():
    ldd = LDD(TVPI, 3, 3)
    f: Function
    f = ldd.true
    i = f.ref
    ldd._incref(f.node)
    j = f.ref
    if j != i + 1:
        raise AssertionError((j, i))
    # avoid errors in `BDD.__dealloc__`
    ldd._decref(f.node, recursive=True)
    del f

cpdef _test_decref():
    ldd = LDD(TVPI, 3, 3)
    f: Function
    f = ldd.true
    i = f.ref
    if i != 2:
        raise AssertionError(i)
    ldd._incref(f.node)
    i = f.ref
    if i != 3:
        raise AssertionError(i)
    ldd._decref(f.node, recursive=True)
    j = f.ref
    if j != i - 1:
        raise AssertionError((j, i))
    del f

cpdef _test_call_dealloc(
        u: Function):
    """Duplicates the code of `Function.__dealloc__`.

    The main purpose of this function is to test the
    exceptions raised in the method `Function.__dealloc__`.

    Exceptions raised in `__dealloc__` are ignored
    (they become messages), and it seems impossible to
    call `__dealloc__` directly (unlike `__del__`),
    so there is no way to assert what exceptions
    are raised in `__dealloc__`.

    This function is the closest thing to testing
    those exceptions.
    """
    self = u
    # the code of `Function.__dealloc__` follows:
    if self._ref < 0:
        raise AssertionError(
            "The lower bound `_ref` on the node's "
            'reference count has value {self._ref}, '
            'which is unexpected and should never happen. '
            'Was the value of `_ref` changed from outside '
            'this instance?')
    assert self._ref >= 0, self._ref
    if self._ref == 0:
        return
    if self.node is NULL:
        raise AssertionError(
            'The attribute `node` is a `NULL` pointer. '
            'This is unexpected and should never happen. '
            'Was the value of `_ref` changed from outside '
            'this instance?')
    # anticipate multiple calls to `__dealloc__`
    self._ref -= 1
    # deref
    Cudd_IterDerefBdd(self.cudd_manager, self.node)
    # avoid future access to deallocated memory
    self.node = NULL
