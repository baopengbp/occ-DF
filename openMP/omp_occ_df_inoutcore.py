#!/usr/bin/env python
# occ_DF
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
occ incore-outcore Density Fitting

Using occ method to make incore-outcore DF faster
(Manzer, S.; Horn, P. R.; Mardirossian, N.; Head-Gordon, M. 
J. Chem. Phys. 2015, 143, 024113.)

openMP: 'MKL_NUM_THREADS=28 OMP_NUM_THREADS=28 python omp_occ_df_inoutcore.py'
'''

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
from pyscf import gto, scf, dft, lib, df
from pyscf.df import addons
lib.logger.TIMER_LEVEL = 0
import outcore_l

mol = gto.M(atom='''
C 2.16778 -0.44918 -0.04589
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
H 2.08462 -1.54360 -0.25842
''',
#            charge=1,
#            spin=1,
            basis='ccpvdz')
print('basis=',mol.basis,'nao',mol.nao)

auxbasis = 'ccpvdz-jk-fit'
print('auxbasis',auxbasis)
# df
mf = scf.RHF(mol).density_fit()
mf.verbose = 4
#mf.with_df.max_memory = 8000
#mf.max_cycle = 0
mf.kernel()
print('reference DF total energy =', mf.e_tot)

mf = scf.RHF(mol).density_fit()
mf.direct_scf = False
#mf.with_df.max_memory = 8000
#mf.max_cycle = 0

auxmol = df.addons.make_auxmol(mol, auxbasis)
# (P|Q)
int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
naux0 = int2c.shape[0]
print('naux0',naux0)
# (ij|P)
int3cj = df.incore.aux_e2(mol, auxmol, 'int3c2e_sph', aosym='s2ij', comp=1).T

def build(self):
    t0 = (time.clock(), time.time())
    log = logger.Logger(self.stdout, self.verbose)

    self.check_sanity()
    self.dump_flags()

    mol = self.mol
    auxmol = self.auxmol = df.addons.make_auxmol(self.mol, self.auxbasis)
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    nao_pair = nao*(nao+1)//2

    max_memory = (self.max_memory - lib.current_memory()[0]) * .8
    int3c = mol._add_suffix('int3c2e')
    int2c = mol._add_suffix('int2c2e')
    if (nao_pair*naux*3*8/1e6 < max_memory and
        not isinstance(self._cderi_to_save, str)):
### replace 3c incore
        self._cderi = int3cj
    else:
        if isinstance(self._cderi_to_save, str):
            cderi = self._cderi_to_save
        else:
            cderi = self._cderi_to_save.name
        if isinstance(self._cderi, str):
            log.warn('Value of _cderi is ignored. DF integrals will be '
                     'saved in file %s .', cderi)
### eri replace using outcore_l
        outcore_l.cholesky_eri(mol, cderi, dataname='j3c',
                             int3c=int3c, int2c=int2c, auxmol=auxmol,
                             max_memory=max_memory, verbose=log)
        if nao_pair*naux*8/1e6 < max_memory:
            with df.addons.load(cderi, 'j3c') as feri:
                cderi = numpy.asarray(feri)
        self._cderi = cderi
        log.timer_debug1('Generate density fitting integrals', *t0)
    return self

df.DF.build = build(mf.with_df)

#@profile
def get_jk(dfobj, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
    t0 = t1 = (time.clock(), time.time())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    assert(with_j or with_k)

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

    if not with_k:
        dmtril = []
        orho = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5
                orho.append(numpy.empty((naux0)))
        b0 = 0
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho = numpy.tensordot(eri1, dmtril[k], axes=([1],[0]))
                    orho[k][b0:b1] = rho
            b0 = b1
        ivj = []
        for k in range(nset):
            ivj.append(scipy.linalg.solve(int2c, orho[k]))
        b0 = 0
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.tensordot(ivj[k][b0:b1], eri1, axes=([0],[0]))
            vk[k] = numpy.zeros((nao,nao))
            b0 = b1
            t1 = log.timer_debug1('jk', *t1)

    elif hasattr(dm, 'mo_coeff'):
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
        orho = []
        obuf1 = []
        obuf2r = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5
                orho.append(numpy.empty((naux0)))
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))
            orbo0.append(numpy.asarray(mo_coeff[k][:,mo_occ[k]>0], order='F'))
            nocc1 = orbo[k].shape[1]
            nocc_pair = nocc1*(nocc1+1)//2
            obuf1.append(numpy.empty((nocc1*naux0, nao)))
            obuf2r.append(numpy.empty((naux0,nocc_pair)))
        buf = numpy.empty((dfobj.blockdim*nao,nao))
        nocc1 = orbo[0].shape[1]
        b0 = 0
        c0 = [0] * nset
        c1 = [0] * nset
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho = numpy.tensordot(eri1, dmtril[k], axes=([1],[0]))
                    orho[k][b0:b1] = rho
                nocc = orbo[k].shape[1]
                c1[k] = c0[k] + naux*nocc
                if nocc > 0:
                    buf1 = buf[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    obuf1[k][c0[k]:c1[k]] = buf1
                    buf2 = lib.dot(buf1, orbo0[k])
                    obuf2r[k][b0:b1] = lib.pack_tril(buf2.reshape(naux,nocc,-1))
                    c0[k] = c1[k]
            b0 = b1 
            t1 = log.timer_debug1('jk', *t1)
        ivj = []
        iokr = []
        for k in range(nset):
            ivj.append(scipy.linalg.solve(int2c, orho[k]))
            iok = scipy.linalg.solve(int2c, obuf2r[k])
            nocc = orbo[k].shape[1]
            iokr.append(lib.unpack_tril(iok).reshape(naux0*nocc,-1))     
        b0 = 0
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.tensordot(ivj[k][b0:b1], eri1, axes=([0],[0]))
            b0 = b1         
        for k in range(nset):
            kiv = lib.dot(iokr[k].T, obuf1[k])
# project iv -> uv
            ovlp = mol.intor_symmetric('int1e_ovlp')
            sc = numpy.dot(ovlp, orbo0[k])
            sck = numpy.dot(sc, kiv)
            kij = lib.einsum('ui,ju->ij', orbo0[k], kiv)
            vk[k] += sck + sck.T - reduce(numpy.dot, (sc, kij, sc.T))
    else:
        dmtril = []
        orho = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5
                orho.append(numpy.empty((naux0)))
        b0 = 0
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
#                    rho = numpy.einsum('px,x->p', eri1, dmtril[k])
                    rho = numpy.tensordot(eri1, dmtril[k], axes=([1],[0]))
                    orho[k][b0:b1] = rho
            b0 = b1
        ivj = []
        for k in range(nset):
            ivj.append(scipy.linalg.solve(int2c, orho[k]))
        b0 = 0
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
#                    vj[k] += numpy.einsum('p,px->x', ivj[b0:b1], eri1)
                    vj[k] += numpy.tensordot(ivj[k][b0:b1], eri1, axes=([0],[0]))

            b0 = b1
            t1 = log.timer_debug1('jk', *t1)
        vk = numpy.zeros(dm_shape)
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = numpy.asarray(vk).reshape(dm_shape)
    logger.timer(dfobj, 'vj and vk', *t0)
    return vj, vk

# Overwrite the default get_jk to apply the new J/K builder
df.df_jk.get_jk = get_jk
mf.verbose = 4
mf.kernel()
print('Approximate HF total energy =', mf.e_tot)
