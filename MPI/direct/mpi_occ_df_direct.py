#!/usr/bin/env python
# occ_DF
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
mpi occ direct Density Fitting

Using occ method to make direct DF faster
(Manzer, S.; Horn, P. R.; Mardirossian, N.; Head-Gordon, M. 
J. Chem. Phys. 2015, 143, 024113.)

'MKL_NUM_THREADS=7 OMP_NUM_THREADS=7 mpirun -np 4 python mpi_occ_df_direct.py'
'''

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


from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

#@profile
def loop(self, blksize=None):
# direct  blocksize
    mol = self.mol
    auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
    if auxmol is None:
        auxmol = make_auxmol(mol, auxbasis)
    int3c='int3c2e'
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
    comp = 1
    aosym = 's2ij'
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

    nx = 0
    nxu = 0
### use 1/10 of the block
    BLKSIZE = min(80, (stop-start)//10+1)  
    for p0, p1 in lib.prange(start,stop, BLKSIZE):
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+p0, mol.nbas+p1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                  aosym, ao_loc, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1
 
'''
# need modify bcast_tagged_array(arr) in mpi4pyscf/tools/mpi.py for very big array to:

def bcast_tagged_array_occdf(arr):
#   'Broadcast big nparray or tagged array.'
    if comm.bcast(not isinstance(arr, numpy.ndarray)):
        return comm.bcast(arr)

    new_arr = bcast(arr)

    if comm.bcast(isinstance(arr, lib.NPArrayWithTag)):
        new_arr = lib.tag_array(new_arr)
        if rank == 0:
            kv = []
            for k, v in arr.__dict__.items():
                kv.append((k, v))
            comm.bcast(kv)
        else:
            kv = comm.bcast(None)
            new_arr.__dict__.update(kv)

        for k, v in kv:
            if v is 'NPARRAY_TO_BCAST':
                new_arr.k = bcast(v)

    if rank != 0:
        arr = new_arr
    return arr
'''

#@profile
@mpi.parallel_call(skip_args=[1])
def get_jk(mol_or_mf, dm, hermi=1):
    '''MPI version of scf.hf.get_jk function'''
    #vj = get_j(mol_or_mf, dm, hermi)
    #vk = get_k(mol_or_mf, dm, hermi)
    with_j=True
    with_k=True
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf
    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(isinstance(dm, str) and dm == 'SKIPPED_ARG')):
        dm = mpi.bcast_tagged_array_occdf(dm)
        #dm = mpi.bcast_tagged_array(dm)
    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
 
    auxbasis = 'weigend'
    mf.auxbasis = auxbasis
    mf.with_df = mf
    mol = mf.mol
    global int2c
# use sttr to calc int2c once
    if not hasattr(dm, 'mo_coeff'):
        auxmol = df.addons.make_auxmol(mol, auxbasis)
# (P|Q)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)

    naux0 = int2c.shape[0]
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

    if hasattr(dm, 'mo_coeff'):
#TODO: test whether dm.mo_coeff matching dm
        mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
        mo_occ   = numpy.asarray(dm.mo_occ)
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
        buf2r = []
        b0 = 0
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                nocc = orbo[k].shape[1]
                if nocc > 0:
                    buf1 = numpy.empty((naux*nocc,nao))
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    buf2r0 = lib.dot(buf1, orbo0[k]).reshape(naux,nocc,-1)
                    if with_j:
                        if b0 == 0: 
                            rho.append(numpy.empty(paux[rank]))
                            nocc_pair = nocc*(nocc+1)//2
                            buf2r.append(numpy.empty((paux[rank],nocc_pair)))
                        rho[k][b0:b1] = numpy.einsum('kii->ki', buf2r0).dot(numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
                    buf2r[k][b0:b1] = lib.pack_tril(buf2r0)
            b0 = b1 

        iokr = []
        orho = []
#        obuf2r = []
        rec = []
        for k in range(nset):
# gather buf2r
            split = numpy.array_split(buf2r[k],mpi.pool.size,axis = 1)
            orho.append(mpi.gather(rho[k]))
            for i in range(mpi.pool.size):
                obuf2r0 = mpi.gather(split[i],root=i)
                if rank == i:
######### last time modify
                    obuf2r = obuf2r0
            obuf2r0 = None
            split = None
# j
            if rank == 0:
                ivj0 = scipy.linalg.solve(int2c, orho[k])
            else:
                ivj0 = None
            rec.append(numpy.empty(paux[rank]))
            comm.Scatterv([ivj0,paux],rec[k],root=0)
# k
            iok = scipy.linalg.solve(int2c, obuf2r)
            for i in range(mpi.pool.size):
                if i == 0:
                    j0 = 0
                else:
                    j0 = pauxz[i-1]
                j1 = pauxz[i]
                iok0 = mpi.gather(iok[j0:j1].reshape(-1,order='F'),root=i)
                if rank == i:
                    iokx = lib.unpack_tril(iok0.reshape((paux[rank],-1),order='F'))
            iok0 = None
            iok  = None
            nocc = orbo[k].shape[1]
            iokr.append(iokx.reshape(paux[rank]*nocc,-1)) 
            iokx = None  
        b0 = 0
        c0 = [0] * nset
        c1 = [0] * nset 
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                nocc = orbo[k].shape[1]
                if nocc > 0:
                    c1[k] = c0[k] + naux*nocc
                    buf1 = numpy.empty((naux*nocc,nao))
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    kiv[k] += lib.dot(iokr[k][c0[k]:c1[k]].T, buf1)
                    if with_j:
                        vj[k] += numpy.dot(rec[k][b0:b1].T, eri1)
                    c0[k] = c1[k]
            b0 = b1     
        for k in range(nset):
            kiv[k] = comm.reduce(kiv[k])

# project iv -> uv
            if rank == 0:
                ovlp = mol.intor_symmetric('int1e_ovlp')
                sc = numpy.dot(ovlp, orbo0[k])
                sck = numpy.dot(sc, kiv[k])
                kij = lib.einsum('ui,ju->ij', orbo0[k], kiv[k])
                vk[k] += sck + sck.T - reduce(numpy.dot, (sc, kij, sc.T))
            else:
                vk[k] = None
            vj[k] = comm.reduce(vj[k])
    else:
        dmtril = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5
        rho = []
        b0 = 0
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    if b0 == 0: rho.append(numpy.empty(paux[rank]))
                    rho[k][b0:b1] = numpy.dot(eri1, dmtril[k])
            b0 = b1

        orho = []
        rec = []
        for k in range(nset):
            orho.append(mpi.gather(rho[k]))
            if rank == 0:
                ivj0 = scipy.linalg.solve(int2c, orho[k])
            else:
                ivj0 = None
            rec.append(numpy.empty(paux[rank]))
            comm.Scatterv([ivj0,paux],rec[k],root=0)
        b0 = 0
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.dot(rec[k][b0:b1].T, eri1)
#                    vj[k] += numpy.tensordot(rec[k], eri1, axes=([0],[0]))
            b0 = b1
        for k in range(nset):
            vj[k] = comm.reduce(vj[k])
        vk = numpy.zeros(dm_shape)
    if rank == 0:
        if with_j: vj = lib.unpack_tril(numpy.asarray(vj), 1).reshape(dm_shape)
        if with_k: vk = numpy.asarray(vk).reshape(dm_shape)
    return vj, vk

@mpi.register_class
class SCF(hf.SCF):

    @lib.with_doc(hf.SCF.get_jk.__doc__)
    def get_jk(self, mol, dm, hermi=1):
        assert(mol is self.mol)
        return get_jk(self, dm, hermi)

    def pack(self):
        return {'verbose': self.verbose,
                'direct_scf_tol': self.direct_scf_tol}
    def unpack_(self, mf_dic):
        self.__dict__.update(mf_dic)
        return self

class RHF(SCF):
    pass

@mpi.register_class
class UHF(uhf.UHF, SCF):

    get_jk = SCF.get_jk

if __name__ == '__main__':

    from mpi4pyscf import scf as mpi_scf

    mol = gto.M(atom='''C 2.16778 -0.44918 -0.04589
C 1.60190 0.33213 -1.23844
C 0.12647 -0.01717 -1.47395
C -0.77327 0.24355 -0.24682
C 1.32708 -0.12314 1.19439
C -0.15779 -0.45451 0.98457
C 3.64976 -0.11557 0.18351
O -0.82190 -0.04249 2.15412
C -2.23720 -0.18449 -0.53693
C -3.19414 0.02978 0.65049
C -2.83291 0.54586 -1.75716
H -0.78551 -0.74689 2.78026
H 3.79206 0.96776 0.40070
H 4.06333 -0.69236 1.04258
H 4.26160 -0.36608 -0.71348
H -2.98219 -0.66492 1.49356
H -3.14394 1.07829 1.02340
H -4.25233 -0.17395 0.36759
H -2.76237 1.65099 -1.63614
H -2.33431 0.26367 -2.71114
H -3.90769 0.29147 -1.90369
H -2.23894 -1.27795 -0.76804
H -0.76619 1.34209 -0.04759
H -0.29859 -1.55818 0.89954
H 1.70030 1.42806 -1.05330
H 2.19423 0.10028 -2.15602
H -0.21377 0.58584 -2.34721
H 0.04969 -1.09183 -1.76338
H 1.71426 -0.69369 2.07216
H 1.43065 0.96079 1.43691
H 2.08462 -1.54360 -0.25842''',
#            charge=-1,
#            spin=1,
            basis='ccpvdz')
    print('basis=',mol.basis,'nao',mol.nao)



    Jtime=time.time()
    mf = mpi_scf.RHF(mol)
#mf.max_cycle = 0
    mf.verbose = 4
#mf.kernel()
    print "Took this long for intg: ", time.time()-Jtime
#mf = mpi_scf.RHF(mol)

    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
#mf.max_cycle = 0
    mf.verbose = 4
#mf.kernel()


    import mpi_occ_df_direct

    Jtime=time.time()
    mf = mpi_occ_df_direct.RHF(mol)
    mf.direct_scf = False
    mf.max_cycle = 20
    mf.verbose = 4
    mf.kernel()
    print "Took this long for Rs: ", time.time()-Jtime




