#!/usr/bin/env python
# DF analytical nuclear gradients 
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
mpi DF analytical nuclear gradients

(see Bostrom, J.; Aquilante, F.; Pedersen, T. B.; Lindh, R. J. Chem. Theory Comput. ?2012, 9, 204.)

'''


import time
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf


from pyscf.scf import hf
from pyscf.scf import jk
from pyscf.scf import _vhf
import sys
import copy
import time
import ctypes
from functools import reduce
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
import numpy
import scipy.linalg
from pyscf import gto, scf, dft, lib
from pyscf.df import addons
from pyscf import df
from pyscf.scf import uhf
import pyscf

import mpi_occ_df_incore


from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank



#@profile
def loop_aux(self, intor='int3c2e_ip1', aosym='s1', comp=3):
# direct  blocksize
    mol = self.mol
    auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
    int3c = intor
    int3c = mol._add_suffix(int3c)
    int3c = gto.moleintor.ascint3(int3c)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]
    naoaux = ao_loc[-1] - nao
    # TODO: Libcint-3.14 and newer version support to compute int3c2e without
    # the opt for the 3rd index.
    #if '3c2e' in int3c:
    #    cintopt = gto.moleintor.make_cintopt(atm, mol._bas, env, int3c)
    #else:
    #    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    segsize = (naoaux+mpi.pool.size-1) // mpi.pool.size
    global pauxz,paux
    aa = 0
    while naoaux-aa > segsize-1:
        aa = 0
        j = 0
        b1 = []
        pauxz = []
        paux = []
        for i in range(auxmol.nbas+1):
            if ao_loc[mol.nbas+i]-nao-aa > segsize and j < mpi.pool.size-1:
                paux.append(ao_loc[mol.nbas+i-1]-nao-aa)
                aa = ao_loc[mol.nbas+i-1]-nao
                pauxz.append(aa)
                b1.append(i-1)
                j += 1
        if naoaux-aa <= segsize:
            b1.append(auxmol.nbas)
            paux.append(naoaux-aa)            
            pauxz.append(ao_loc[-1])
        segsize += 1
    stop = b1[rank]    
    if rank ==0:
        start = 0
    else:
       start = b1[rank-1]

    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+start, mol.nbas+stop)
    buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                  aosym, ao_loc, cintopt)
    return buf

#@profile
@mpi.parallel_call(skip_args=[1])
def get_jkgrd(mol_or_mf, dm, mo_coeff=None, mo_occ=None):
    '''MPI version of scf.hf.get_jk function'''
    if rank ==0: print('uuuu00',lib.current_memory()[0])

    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf
    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(isinstance(dm, str) and dm == 'SKIPPED_ARG')):
        dm = mpi.bcast_tagged_array_occdf(dm)
    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()

    mf.with_df = mf
    mol = mf.mol

    auxbasis = mf.auxbasis
    auxbasis = comm.bcast(auxbasis)
    mf.auxbasis = comm.bcast(mf.auxbasis)
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    if rank==0:
        print('number of AOs', nao)
        print('number of auxiliary basis functions', naux)
# (d/dX i,j|P)
#    int3c_e1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', aosym='s1', comp=3)
    int3c_e1 = loop_aux(mf, intor='int3c2e_ip1', aosym='s1', comp=3)
# (i,j|d/dX P)
#    int3c_e2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2', aosym='s1', comp=3)
    int3c_e2 = loop_aux(mf, intor='int3c2e_ip2', aosym='s1', comp=3)
# (d/dX P|Q)
    int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

    if 0==0:
        nmo = mo_occ.shape[-1]
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
            mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
            assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
            mo_occ = numpy.vstack((mo_occa, mo_occb))

        dmtril = []
        orbo = []
        orbo0 = []
        kiv = []
        for k in range(nset):
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))
            orbo0.append(numpy.asarray(mo_coeff[k][:,mo_occ[k]>0], order='F'))
            nocc = orbo[k].shape[1]
            kiv.append(numpy.zeros((nocc,nao)))
        rho = []
        split = []
        buf1 = []

        iokr = mpi_occ_df_incore.iokr
        rec = mpi_occ_df_incore.rec
        k = 0

        dmtril = []
        for k in range(nset):
            dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
            i = numpy.arange(nao)
            dmtril[k][i*(i+1)//2+i] *= .5
        tmp0 = int3c_e2.swapaxes(1,3).reshape(-1,nao*nao)
        tmp  = numpy.dot(tmp0, dm.reshape(-1))      
        tmp0 = numpy.einsum('xp,p->xp', tmp.reshape(3,-1),rec[k])
        ec1_3cp = mpi.gather(tmp0.reshape(-1,order='F')).reshape((3,-1),order='F')
        ec1_3cu_b = numpy.einsum('xuvp,p,uv->xu', int3c_e1,rec[k],dm)
        ec1_3cu = comm.reduce(ec1_3cu_b)
        if rank == 0:
            j0 = 0
        else:
            j0 = pauxz[rank-1]
        j1 = pauxz[rank]
        tmp0 = int2c_e1[:,:,j0:j1].dot(rec[k])
        tmp = comm.allreduce(tmp0)        
        tmp0 = numpy.einsum('xp,p->xp', tmp[:,j0:j1],rec[k])
        ec1_2c = mpi.gather(tmp0.reshape(-1,order='F')).reshape((3,-1),order='F')

        coeff3mo = numpy.sqrt(2.0e0)*(iokr[k].reshape(-1,nocc,nocc))
        tmp =  numpy.tensordot(coeff3mo, orbo[k], axes=([2],[1]))
        coeffb =  numpy.tensordot(tmp, orbo[k], axes=([1],[1]))
        ex1_3cp_b = numpy.einsum('puv,xuvp->xp', coeffb, int3c_e2)
        ex1_3cp = mpi.gather(ex1_3cp_b.reshape(-1,order='F')).reshape((3,-1),order='F')
        ex1_3cu_b = numpy.einsum('puv,xuvp->xu', coeffb, int3c_e1)
        ex1_3cu = comm.reduce(ex1_3cu_b)
        tmp0 = mpi.allgather(coeff3mo)
        tmp = numpy.tensordot(tmp0, coeff3mo, axes=([1,2],[1,2]))
        ex1_2c_b = numpy.einsum('xpq,pq->xp', int2c_e1[:,:,j0:j1], tmp)
        ex1_2c = comm.reduce(ex1_2c_b)
        eaux1p = numpy.empty((3,naux))
        eaux1u = numpy.empty((3,naux))
        if rank == 0: 
            eaux1p = -0.5*(2*ec1_3cp - 2*ec1_2c - 0.5*(2*ex1_3cp - 2*ex1_2c))
            eaux1u = -0.5*(4*ec1_3cu - 0.5*(4*ex1_3cu))
    return eaux1p,eaux1u

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (time.clock(), time.time())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
#    vhf = mf_grad.get_veff(mol, dm0)
    vhfdfp, vhfdfu = get_jkgrd(mf, dm0, mo_coeff, mo_occ)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    auxbasis = mf.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    aux_offset = auxmol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        aux0, aux1 = aux_offset[ia][2:]
#        print('atom %d %s, shell range %s:%s, AO range %s:%s, aux-AO range %s:%s' %
#             (ia, mol.atom_symbol(ia), shl0, shl1, p0, p1, aux0, aux1))
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
# nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
#        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2

#        vhfgrd = numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
#        de[k] += vhfgrd

        vhfdfgrd = numpy.einsum('xp->x', vhfdfp[:,aux0:aux1])
        vhfdfgrd += numpy.einsum('xu->x', vhfdfu[:,p0:p1])
        de[k] += vhfdfgrd

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        if mf_grad.grid_response: # Only effective in DFT gradients
            de[k] += vhf.exc1_grid[ia]
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de, atmlst)
        if mf_grad.grid_response:
            log.debug('grids response contributions')
            _write(log, mol, vhf.exc1_grid[atmlst], atmlst)
            log.debug1('sum(de) %s', vhf.exc1_grid.sum(axis=0))
    return de

def _write(dev, mol, de, atmlst):
    if atmlst is None:
        atmlst = range(mol.natm)
    dev.stdout.write('         x                y                z\n')
    for k, ia in enumerate(atmlst):
        dev.stdout.write('%d %s  %15.10f  %15.10f  %15.10f\n' %
                         (ia, mol.atom_symbol(ia), de[k,0], de[k,1], de[k,2]))


def grad_nuc(mol, atmlst=None):
    gs = numpy.zeros((mol.natm,3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = numpy.sqrt(numpy.dot(r1-r2,r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    if atmlst is not None:
        gs = gs[atmlst]
    return gs


def get_hcore(mol):
    '''Part of the nuclear gradients of core Hamiltonian'''
    h = mol.intor('int1e_ipkin', comp=3)
    if mol._pseudo:
        NotImplementedError('Nuclear gradients for GTH PP')
    else:
        h+= mol.intor('int1e_ipnuc', comp=3)
    if mol.has_ecp():
        h += mol.intor('ECPscalar_ipnuc', comp=3)
    return -h

def hcore_generator(mf, mol=None):
    if mol is None: mol = mf.mol
    with_x2c = getattr(mf.base, 'with_x2c', None)
    if with_x2c:
        hcore_deriv = with_x2c.hcore_deriv_generator(deriv=1)
    else:
        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        aoslices = mol.aoslice_by_atom()
        h1 = mf.get_hcore(mol)
        def hcore_deriv(atm_id):
            shl0, shl1, p0, p1 = aoslices[atm_id]
            with mol.with_rinv_as_nucleus(atm_id):
                vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                vrinv *= -mol.atom_charge(atm_id)
                if with_ecp and atm_id in ecp_atoms:
                    vrinv += mol.intor('ECPscalar_iprinv', comp=3)
            vrinv[:,p0:p1] += h1[:,p0:p1]
            return vrinv + vrinv.transpose(0,2,1)
    return hcore_deriv

def get_ovlp(mol):
    return -mol.intor('int1e_ipovlp', comp=3)

def get_jk(mol, dm):
    '''J = ((-nabla i) j| kl) D_lk
    K = ((-nabla i) j| kl) D_jk
    '''
    intor = mol._add_suffix('int2e_ip1')
    vj, vk = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
                               's2kl', # ip1_sph has k>=l,
                               ('lk->s1ij', 'jk->s1il'),
                               dm, 3, # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
    return -vj, -vk

def get_veff(mf_grad, mol, dm):
    '''NR Hartree-Fock Coulomb repulsion'''
    vj, vk = mf_grad.get_jk(mol, dm)
    return vj - vk * .5

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    mo0 = mo_coeff[:,mo_occ>0]
    mo0e = mo0 * (mo_energy[mo_occ>0] * mo_occ[mo_occ>0])
    return numpy.dot(mo0e, mo0.T.conj())


def as_scanner(mf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
    >>> e_tot, grad = hf_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    >>> e_tot, grad = hf_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)

    class SCF_GradScanner(mf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mf_scanner = self.base
            e_tot = mf_scanner(mol)
            self.mol = mol
            de = self.kernel(**kwargs)
            return e_tot, de
    return SCF_GradScanner(mf_grad)


class Gradients(lib.StreamObject):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.base = scf_method
        self.max_memory = self.mol.max_memory
# This parameter has no effects for HF gradients. Add this attribute so that
# the kernel function can be reused in the DFT gradients code.
        self.grid_response = False

        self.atmlst = None
        self.de = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        if not self.base.converged:
            log.warn('Ground state SCF not converged')
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    hcore_generator = hcore_generator

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return get_ovlp(mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        cpu0 = (time.clock(), time.time())
        #TODO: direct_scf opt
        vj, vk = get_jk(mol, dm)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        intor = mol._add_suffix('int2e_ip1')
        return -_vhf.direct_mapdm(intor, 's2kl', 'lk->s1ij', dm, 3,
                                  mol._atm, mol._bas, mol._env)

    def get_k(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        intor = mol._add_suffix('int2e_ip1')
        return -_vhf.direct_mapdm(intor, 's2kl', 'jk->s1il', dm, 3,
                                  mol._atm, mol._bas, mol._env)

    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        return grad_nuc(mol, atmlst)

    def grad(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        return self.kernel(mo_energy, mo_coeff, mo_occ, atmlst)
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (time.clock(), time.time())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        logger.timer(self, 'SCF gradients', *cput0)
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            _write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {'He': 'ccpvdz'}
    mol.build()
    method = scf.RHF(mol)
    method.scf()
    g = Gradients(method)
    print(g.grad())

    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o.basis = {'H': '631g',
                 'O': '631g',}
    h2o.build()
    rhf = scf.RHF(h2o)
    rhf.conv_tol = 1e-14
    e0 = rhf.scf()
    g = Gradients(rhf)
    print(g.grad())
#[[ 0   0               -2.41134256e-02]
# [ 0   4.39690522e-03   1.20567128e-02]
# [ 0  -4.39690522e-03   1.20567128e-02]]

    rhf = scf.RHF(h2o).x2c()
    rhf.conv_tol = 1e-14
    e0 = rhf.scf()
    g = Gradients(rhf)
    print(g.grad())
#[[ 0   0               -2.40286232e-02]
# [ 0   4.27908498e-03   1.20143116e-02]
# [ 0  -4.27908498e-03   1.20143116e-02]]
