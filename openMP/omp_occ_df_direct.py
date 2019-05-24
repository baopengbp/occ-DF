#!/usr/bin/env python
# occ_DF
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
occ direct Density Fitting

Using occ method to make direct DF faster
(Manzer, S.; Horn, P. R.; Mardirossian, N.; Head-Gordon, M. 
J. Chem. Phys. 2015, 143, 024113.)

openMP: 'MKL_NUM_THREADS=28 OMP_NUM_THREADS=28 python omp_occ_df_direct.py'
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

mol = gto.M(atom='''
H 0 0 0
H 0 0 0.8
''',
#            charge=1,
#            spin=1,
            basis='ccpvdz')
print('basis=',mol.basis,'nao',mol.nao)

auxbasis = 'ccpvdz-jk-fit'
# df
mf = scf.RHF(mol).density_fit()
mf.verbose = 4
mf.with_df.max_memory = 120000
#mf.init_guess = 'atom'
#mf.max_cycle = 0
mf.kernel()
print('reference DF total energy =', mf.e_tot)

#occRIK
mf = scf.RHF(mol).density_fit()
mf.direct_scf = False
mf.with_df.max_memory = 60000
#mf.max_cycle = 0
#mf.init_guess = 'atom'

auxmol = df.addons.make_auxmol(mol, auxbasis)
# (P|Q)
int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
naux0 = int2c.shape[0]

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

    if blksize is None:
        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
        blksize = min(int(max_memory*1e6/48/nao/(nao+1)), 80)
#        print('blksize1',blksize,int(max_memory*1e6/48/nao/(nao+1)),max_memory,self.max_memory)
    comp = 1
    aosym = 's2ij'
    for b0, b1 in self.prange(0, auxmol.nbas, blksize):
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+b0, mol.nbas+b1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                      aosym, ao_loc, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1

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
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            print('slice-naux',naux)
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho = numpy.dot(eri1, dmtril[k])
                    orho[k][b0:b1] = rho
            b0 = b1

        ivj = []
        for k in range(nset):
            ivj.append(scipy.linalg.solve(int2c, orho[k]))
        b0 = 0
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.tensordot(ivj[k][b0:b1], eri1, axes=([0],[0]))
            b0 = b1
            t1 = log.timer_debug1('jk', *t1)
        vk = numpy.zeros(dm_shape)

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
        obuf2r = []
        kiv = []
        for k in range(nset):
            orho.append(numpy.empty((naux0)))
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))
            orbo0.append(numpy.asarray(mo_coeff[k][:,mo_occ[k]>0], order='F'))
            nocc1 = orbo[k].shape[1]
            nocc_pair = nocc1*(nocc1+1)//2
            obuf2r.append(numpy.empty((naux0,nocc_pair)))
            kiv.append(numpy.zeros((nocc1,nao)))
        nocc1 = orbo[0].shape[1]
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
                    buf2r = lib.dot(buf1, orbo0[k]).reshape(naux,nocc,-1)

                    if with_j:
                        rho = numpy.einsum('kii->ki', buf2r).dot(numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
                        orho[k][b0:b1] = rho
                    obuf2r[k][b0:b1] = lib.pack_tril(buf2r)
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
        c0 = [0] * nset
        c1 = [0] * nset
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                nocc = orbo[k].shape[1]
                c1[k] = c0[k] + naux*nocc
                if nocc > 0:
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
                        vj[k] += numpy.dot(ivj[k][b0:b1].T, eri1)
                    c0[k] = c1[k]
            b0 = b1         
        for k in range(nset):
# project iv -> uv
            ovlp = mol.intor_symmetric('int1e_ovlp')
            sc = numpy.dot(ovlp, orbo0[k])
            sck = numpy.dot(sc, kiv[k])
            kij = lib.einsum('ui,ju->ij', orbo0[k], kiv[k])
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
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            print('slice-naux',naux)
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho = numpy.dot(eri1, dmtril[k])
                    orho[k][b0:b1] = rho
            b0 = b1

        ivj = []
        for k in range(nset):
            ivj.append(scipy.linalg.solve(int2c, orho[k]))
#            print('www',ivj[k],ivj[k])
#            exit()
        b0 = 0
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.tensordot(ivj[k][b0:b1], eri1, axes=([0],[0]))
            b0 = b1
            t1 = log.timer_debug1('jk', *t1)
        vk = numpy.zeros(dm_shape)
        print('wwwwq',vj[0])
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = numpy.asarray(vk).reshape(dm_shape)
    logger.timer(dfobj, 'vj and vk', *t0)
    return vj, vk

# Overwrite the default get_jk to apply the new J/K builder
df.df_jk.get_jk = get_jk
mf.verbose = 4
mf.kernel()
print('Approximate HF total energy =', mf.e_tot)
