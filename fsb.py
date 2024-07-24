# author: Lea Harscouet lea.harscouet@physics.ox.ac.uk
# license: ?

'''Pymaster extension template

Functions
---------

_reduce2
get_filters


Classes
---------

FSB
    Methods
    ---------
    __init__
    return_wkspace
    filtered_sq_fields
    get_cls_field
    get_fsb
    get_gauss_cov
    _get_n222_term
    get_n222_cov
    _get_general_fsb
    get_n32_cov
    get_full_cov


'''

######################################################

import pymaster as nmt 
import numpy as np
import healpy as hp 


def _reduce2(bigm):
    """
    Reduces the dimensionality of an array by 2 along its first two axes.

    Arguments
    ----------
    bigm : array with shape (i, j, k, ...)

    Returns
    ----------
    An array of shape (k, ...)

    """
    minus1 = np.hstack(tuple(l for l in bigm)); print(minus1.shape)
    minus2 = np.hstack(tuple(l for l in minus1)); print(minus2.shape)
    return minus2

def get_filters(nbands, nside):
    """
    Linearly divides the ell range (2; 3*nside-1)
    into nbands filters of equal size.

    Arguments
    ----------
    nbands : int
        number of filters
    nside : int
        resolution of the healpy map used

    Returns
    ----------
    A binary array of shape (nbands, 3*nside)
    
    """
    start=2
    nell = (3*nside-1) - start
    dell = nell//nbands
    fls = np.zeros([nbands, 3*nside]) 
    for i in range(nbands):
        fls[i, i*dell+start:(i+1)*dell+start] = 1.0
    return fls




class FSB():

    def __init__(self, map1, mask1, filters, rmask=None, map2=None, mask2=None, w_fsb=None, ells_per_bin=10, niter=3):

        """
        Arguments
        ----------
        map1 : array
            healpy map of size (12*nside**2)
        filters : array
            a binary array of shape (nbands, 3*nside)
        mask1 : array
            a mask of the same size as map1
        rmask : array 
            a mask of the same size as map1, which will be
            applied after squaring the field
        map2 : array, optional
            healpy map of the same size as map1
        mask2 : array
            a mask of the same size as map2
        ells_per_bin : int (default = 10), optional
            the number of ells in each bin in the binning scheme.
            will only be used if wksp is not given.
        niter : int (default = 3)
            the number of iteration used to compute `hp.map2alm`
            and the `NmtField` objects
  
        """

        self.map1 = map1
        self.mask1 = mask1
        self.filters = filters
        self.rmask = rmask
        self.map2 = map2
        self.mask2 = mask2
        self.ells_per_bin = ells_per_bin
        self.niter = niter

        self.npix = len(self.map1) 
        self.nside = hp.npix2nside(self.npix)

        if self.mask1 is None:
            self.mask1 = np.ones(self.npix) 
        if self.rmask is None:
            self.rmask = self.mask1
        if self.map2 is None:
            self.map2 = self.map1
        if self.mask2 is None:
            self.mask2 = self.mask1

        
        self.w_fsb = w_fsb 
        if self.w_fsb is None: # better if not an argument actually
            self.w_fsb = self.return_wkspace(self.rmask, self.mask2, self.ells_per_bin)
        self.w_cls = self.return_wkspace(self.mask1, self.mask1, self.ells_per_bin)

        self.fsky_fsb = np.mean(self.rmask*self.mask2)
        self.fsky_cls = np.mean(self.mask1*self.mask2)

        self.nbands = len(self.filters)

        self.fsb_binned = self.get_fsb(self.w_fsb)
        self.fsb_unbinned = self.get_fsb()

        field1 = nmt.NmtField(self.mask1, [self.map1], masked_on_input=False, n_iter=self.niter)
        self.cls_binned = self.get_cls_field(np.array([field1]), self.mask1, wksp=self.w_cls)
        self.cls_unbinned = self.get_cls_field(np.array([field1]), self.mask1)
        self.cls_fsq_unbinned = self.get_cls_field(self.f1s, self.mask1) # not sure about generalisation to 2 masks

        # maybe do binning here?
        self.bb = nmt.NmtBin.from_lmax_linear(3*self.nside-1, self.ells_per_bin)
        self.b = len(self.bb.get_effective_ells())
        self.bins = get_filters(self.b, self.nside) # corresponding filters for the bins (for generalized fsb)



    def return_wkspace(self, mask1, mask2, lpb): 

        """
        Creates an `NmtWorkspace` object from mask(s) and
        a binning scheme.

        Arguments
        ----------
        mask1 : array
            healpy map of size (12*nside**2)
        mask2 : array
            a healpy map of the same size as mask1
        ells_per_bin : int (default = 10)
            the number of ells in each bin in the binning scheme.

        Returns
        ----------
        A `NmtWorkspace` object corresponding to the
        given mask(s) and binning scheme.
        
        """
            
        fmask1 = nmt.NmtField(mask1, None, spin=0)
        fmask2 = nmt.NmtField(mask2, None, spin=0)

        if lpb==1: # not sure still relevant?
            ells = np.arange(3*self.nside)
            b = nmt.NmtBin(bpws=ells, ells=ells, weights=np.ones(len(ells)))
        else:
            b = nmt.NmtBin.from_lmax_linear(3*self.nside-1, lpb)
        w12 = nmt.NmtWorkspace()
        w12.compute_coupling_matrix(fmask1, fmask2, b) # what about remask though?

        return w12 


    def filtered_sq_fields(self): # generalize for generalized FSB

        """
        Creates NmtField objects for each filtered-squared map.

        Returns
        ----------
        An array containing as many NmtField objects as 
        there are filters.
        
        """ 

        mask1_bin = self.mask1>0   
        map1 = self.map1*mask1_bin       

        alm1 = hp.map2alm(map1, iter=self.niter) # , lmax=3*self.nside-1
        
        mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm1, fl), self.nside, lmax=3*self.nside-1)**2 for fl in self.filters])   
        
        mp_filt_sq = np.array([(m-np.mean(m[mask1_bin==1]))*mask1_bin for m in mp_filt_sq]) # -mean, remasking (binary)

        f1sq = [nmt.NmtField(self.rmask, [m], masked_on_input=False, n_iter=self.niter) for m in mp_filt_sq] 
        
        return np.array(f1sq)
    
    

    def get_cls_field(self, field1, mask1, field2=None, mask2=None, wksp=None):

        """
        Computes the power spectra of the given field(s).
        Handles the following cases:
            - if field1 is a single field, will return 
            its auto-power spectrum
            - if field1 is several fields, will return 
            the corresponding cross-power spectra
            - if field1 and field2 are both single fields,
            will return their cross-power spectrum.
            - if both field1 and field2 are several fields,
            will return their cross-power spectra. (NOT SURE THIS WORKS)

        Arguments
        ----------
        field1 : `NmtField` object or array of `NmtField` objects
        mask1 : array
            a mask of shape corresponding to the individual field1
        field2 : `NmtField` object or array of `NmtField` objects,
        optional
        mask2 : array, optional
            a mask of shape corresponding to the individual field2
        wksp : `NmtWorkspace` object, optional
            a workspace to compute the mode coupling and binning of
            the power spectra. If not given, it will be assumed that
            no binning is required, and the fsky correction is applied
            to the unbinned cls.

        Returns
        ----------
        An array of the cls.
            
        """

        if field2 is None:
            field2 = field1
            same = True
        else:
            same = False
        if mask2 is None:
            mask2 = mask1


        fsky = np.mean(mask1*mask2)

        if field1.shape[0]>1: # several fields as input

            if wksp is None: # auto power spectra, unbinned

                claa = np.zeros((len(field1), len(field2), 3*self.nside)) # ?

                if same is True: # field2 is None: # this loop wont work because field2 = field1 now
                    for n in range(len(field1)):
                        for m in range(n, len(field2)): # TODO: if field2!=field1, should not start from n?
                            cross = nmt.compute_coupled_cell(field1[n], field2[m])[0] / fsky
                            claa[n, m] = cross; claa[m, n] = cross
                else:
                    for n in range(len(field1)):
                        for m in range(len(field2)): # if field2!=field1, should not start from n?
                            claa[n, m] = nmt.compute_coupled_cell(field1[n], field2[m])[0] / fsky
                    if len(field2)==1:
                        claa = claa[:, 0, :] # get rid of extra layer
                                    
            else: # cross power spectra, binned
                claa = np.array([wksp.decouple_cell(nmt.compute_coupled_cell(fi, fi))[0] for fi in field1])

            return claa
        
        else: # one field as input, inside a np.array
            # need to unpack that additional layer.... but what about field2?

            if wksp is None: 
                clbb = nmt.compute_coupled_cell(field1[0], field2[0])[0] / fsky

            else:
                clbb = wksp.decouple_cell(nmt.compute_coupled_cell(field1[0], field2[0]))[0]

            return clbb



    def get_fsb(self, wksp=None):

        """
        Computes the FSB of the class field(s)
        for a set of filters. if map1 and map2 are different, 
        only map1 will be filtered.
        
        Arguments
        ----------
        
        wksp12 : `NmtWorkspace` object, optional
            a workspace to compute the mode coupling and binning of
            the power spectra. if not given, it will be assumed that
            no binning is required, and the fsky correction is applied
            to the unbinned FSB.
        
        Returns
        ----------
        An array of the FSBs.
            
        """
        
        self.f1s = self.filtered_sq_fields()
        f2 = nmt.NmtField(self.mask2, [self.map2], n_iter=self.niter)

        fsb = self.get_cls_field(self.f1s, self.rmask, field2=np.array([f2]), mask2=self.mask2, wksp=wksp) #, cov=self.cov
        
        return fsb
    


    def get_gauss_cov(self):

        """
        Computes the gaussian-limit approximation
        of the FSB+Cl covariance. 
        
        Returns
        ----------
        A FSB+Cl covariance matrix.
        Should be of size ((nbands+1)*ndatapoints, (nbands+1)*ndatapoints)
        if the same binning is used for both FSBs and Cls.
            
        """

        fmask_r = nmt.NmtField(self.rmask, None, spin=0)
        fmask = nmt.NmtField(self.mask2, None, spin=0)
        cw = nmt.NmtCovarianceWorkspace()
        cw.compute_coupling_coefficients(fmask, fmask_r, fmask, fmask_r)

        # w_fsb = self.return_wkspace(rmask, mask2=mask, ells_per_bin=self.ells_per_bin)
        # w_cls = self.return_wkspace(mask, ells_per_bin=self.ells_per_bin)
        # TODO: check where to use w_cls when rmask!= mask

        gauss_cov = np.zeros((self.nbands+1, self.nbands+1, self.b, self.b)) 

        fsbs_cls = np.zeros((self.nbands+1, self.fsb_unbinned.shape[1]))
        fsbs_cls[:self.nbands] = self.fsb_unbinned
        fsbs_cls[-1] = self.cls_unbinned # cls

        # FSB-FSB

        for n in range(len(fsbs_cls)):
            for m in range(n, len(fsbs_cls)):

                clad = fsbs_cls[n]
                clbc = fsbs_cls[m]
                clbd = fsbs_cls[-1] # cls
                
                if m==self.nbands:
                    clac = clad 
                    clbc = clbd
                else:
                    clac = self.cls_fsq_unbinned[n, m] # cls_fsq[n, m]

                if n==self.nbands:
                    clac = fsbs_cls[-1] # cls 
                    clad = fsbs_cls[-1] # cls
                    clbc = fsbs_cls[-1] # cls
                    
                covij_fsb = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [clac], [clad], [clbc], [clbd], self.w_fsb)
                gauss_cov[n, m] = covij_fsb
                gauss_cov[m, n] = covij_fsb
        
        self.gauss_cov = gauss_cov

        return self.gauss_cov



    def _get_n222_term(self, cls, filter):

        """
        Computes the N222 term for each FSB.

        Arguments
        ----------
        cls : array 
            an array containing the power spectrum of the
            original map (before filtering/squaring)
        filter : array
            a single binary array of shape (3*nside, )

        Returns
        ----------
        The not yet mask-corrected N222 term in a
        block matrix of size (len(cls), len(cls)).
        """

        cls_f = cls*filter

        lmax = len(cls_f)-1 
        beam = np.ones_like(cls_f)
        ls = np.arange(lmax+1)
        bin = nmt.NmtBin.from_lmax_linear(lmax, 1) # could use general binning instead

        w = nmt.nmtlib.comp_coupling_matrix(0, 0, lmax, lmax, 0, 0, 0, 0, beam, beam, cls_f, bin.bin, 0, -1, -1, -1)
        sum_l1 = nmt.nmtlib.get_mcm(w, (lmax+1)**2).reshape([lmax+1, lmax+1])
        sum_l1 /= (2*ls+1)[None, :]
        cov = sum_l1 * 4 * np.outer(cls_f, cls_f)
        nmt.nmtlib.workspace_free(w)

        return cov

    def get_n222_cov(self):

        """
        Computes the mask-corrected, binned N222
        terms for all FSBs.

        Returns
        ----------
        The binned mask-corrected N222 term in an
        array of dimensions (nbands+1, nbands+1, nbins).
        """

        n222 = np.zeros((len(self.filters)+1, len(self.filters)+1, self.b, self.b))

        for nb in range(self.nbands):
            n222ub = self._get_n222_term(self.cls_unbinned, self.filters[nb])
            n222_bin1 = np.array([self.bb.bin_cell(row) for row in n222ub])
            n222_final = np.array([self.bb.bin_cell(col) for col in n222_bin1.T]).T
            n222[nb, nb] += n222_final 

        return n222/self.fsky_cls


    def _get_general_fsb(self, filters1, filters2):

        """
        Computes the generalized FSB for 
        two sets of filters.

        Arguments
        ----------
        filters1 : array
            a binary array of shape (nbands1, 3*nside)
        filters2 : array 
            a binary array of shape (nbands2, 3*nside)
        
        Returns
        ----------
        The binned generalized FSBs in an array
        of shape (nbands1, nbands2, 3*nside).
        """

        alms = hp.map2alm(self.map1)
        maps_F = np.array([hp.alm2map(hp.almxfl(alms, fl), self.nside) for fl in filters1])
        maps_B = np.array([hp.alm2map(hp.almxfl(alms, bl), self.nside) for bl in filters2])

        self.cls_FBxOG = np.zeros((len(filters1), len(filters2), len(filters1[0])))

        for f in range(len(filters1)):
            print(f)
            for b in range(len(filters2)):
                print('\t', b)
                map_FB = maps_F[f]*maps_B[b]
                map_FB = (map_FB-np.mean(map_FB[self.rmask==1]))*self.rmask # - mean and remasking (only works for rmask binary)
                self.cls_FBxOG[f, b] = hp.anafast(map_FB, self.map2) / self.fsky_fsb

        return self.cls_FBxOG


    def get_n32_cov(self, filters1, filters2):

        """
        Computes the mask-corrected, binned N32
        terms for all FSBs x Cls.

        Arguments
        ----------
        filters1 : array
            a binary array of shape (nbands1, 3*nside)
        filters2 : array 
            a binary array of shape (nbands2, 3*nside)

        Returns
        ----------
        The binned mask-corrected N32 term in an
        array of dimensions (nbands+1, nbands+1, nbins).
        (if we assume filters2 to be the equivalent of 
        the binning scheme.)
        """

        genfsb = self._get_general_fsb(filters1, filters2)

        cls = hp.anafast(self.map1) / self.fsky_cls # not sure if this should be map1 or map2

        # bin general fsb
        binned_FSB_Ll1l2 = np.zeros((len(filters1), len(filters2), self.b)) # i think not nbands, nbins, nbins necessarily (depends on len of filters2)
        for n in range(len(filters1)):
            for b in range(len(filters2)):
                binned_FSB_Ll1l2[n, b] = self.bb.bin_cell(genfsb[n, b])

        # bin filtered cls
        cls_f = np.array([cls*fl for fl in filters1])
        un = np.array([self.bb.bin_cell(e) for e in cls_f]) 
        deux = np.array([np.repeat([e], len(filters2), axis=0).T for e in un])

        fsb_gen = np.zeros((len(filters1)+1, len(filters1)+1, len(filters2), self.b))
        cl_filter = np.zeros((len(filters1)+1, len(filters1)+1, len(filters2), self.b))

        fsb_gen[len(filters1), :len(filters1)] = binned_FSB_Ll1l2
        cl_filter[len(filters1), :len(filters1)] = deux

        n32 = cl_filter*fsb_gen / ( np.array([(2*self.bb.get_effective_ells()+1)]).T * np.pi * self.fsky_cls) 

        for i in range(self.nbands): # make it symmetric
            n32[i, self.nbands] = n32[self.nbands, i].T

        return n32
    
    def get_full_cov(self, insquares=False):

        """
        Adds the mask-corrected, binned gaussian-
        limit covariance and its main mask-corrected, 
        binned non-gaussian contributions. 

        Returns
        ----------
        An array of dimensions ((nbands+1)*nbins, (nbands+1)*nbins),
        or if insquares set to True, an array of dimensions 
        ((nbands+1), (nbands+1), nbins, nbins).
        """
        self.full_cov_large = self.gauss_cov + self.get_n222_cov() + self.get_n32_cov(self.filters, self.bins)
        self.full_cov = _reduce2(self.full_cov_large)
        
        if insquares is True:
            return self.full_cov_large
        else:
            return self.full_cov
    




