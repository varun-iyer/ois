"""
    Optimal Image Subtraction (OIS) module
    --------------------------------------

    A collection of tools to perform optimal image differencing
    for the Transient Optical Robotic Observatory of the South (TOROS).

    ### Usage example (from python):

        >>> import ois
        >>> conv_image, optimalKernel, background =
            ois.optimalkernelandbkg(image, referenceImage)

    (conv_image is the least square optimal approximation to image)

    See optimalkernelandbkg docstring for more options.

    ### Command line arguments:
    * -h, --help: Prints this help and exits.
    * -v, --version: Prints version information and exits.

    (c) Martin Beroiz

    email: <martinberoiz@gmail.com>

    University of Texas at San Antonio
"""

__version__ = '0.2a1'

import numpy as np
from scipy import signal
from scipy import ndimage


class SubtractionStrategy:

    def __init__(self, image, refimage, kernelshape, bkg_degree):
        self.image = image
        self.refimage = refimage
        self.h, self.w = image.shape
        self.image_data, self.refimage_data, self.badpixmask =\
            self.separate_data_mask()

        self.k_shape = kernelshape
        self.k_side = kernelshape[0]

    def has_mask(self, image):
        is_masked_array = isinstance(image, np.ma.MaskedArray)
        if is_masked_array and isinstance(image.mask, np.ndarray):
            return True
        return False

    def separate_data_mask(self):
        def ret_data(image):
            if isinstance(image, np.ma.MaskedArray):
                image_data = image.data
            else:
                image_data = image
            return image_data
        badpixmask = None
        if self.has_mask(self.refimage):
            badpixmask = ndimage.binary_dilation(
                self.refimage.mask.astype('uint8'),
                structure=np.ones(self.k_shape)).astype('bool')
            if self.has_mask(self.image):
                badpixmask += self.image.mask
        elif self.has_mask(self.image):
            badpixmask = self.image.mask
        return ret_data(self.image), ret_data(self.refimage), badpixmask

    def coeffstobackground(self, coeffs):
        bkgdeg = int(-1.5 + 0.5 * np.sqrt(9 + 8 * (len(coeffs) - 1)))
        h, w = self.h, self.w
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(bkgdeg + 1)]
        allys = [pow(y, i) for i in range(bkgdeg + 1)]
        mybkg = np.zeros(self.image.shape)
        ind = 0
        for i, anX in enumerate(allxs):
            for aY in allys[:bkgdeg + 1 - i]:
                mybkg += coeffs[ind] * anX * aY
                ind += 1
        return mybkg

    def get_cmatrices_background(self):
        h, w = self.refimage.shape
        y, x = np.mgrid[:h, :w]
        allxs = [pow(x, i) for i in range(self.bkg_degree + 1)]
        allys = [pow(y, i) for i in range(self.bkg_degree + 1)]
        bkg_c = [anX * aY for i, anX in enumerate(allxs)
                 for aY in allys[:self.bkg_degree + 1 - i]]
        return bkg_c


class AlardLuptonStrategy(SubtractionStrategy):

    def __init__(self, image, refimage, kernelshape, bkg_degree, gausslist):
        self.super.__init__(image, refimage, kernelshape, bkg_degree)
        if gausslist is None:
            self.gausslist = [{}]
        else:
            self.gausslist = gausslist
        self.clean_gausslist()

    def gauss(self, center, sx, sy):
        h, w = self.k_shape
        x0, y0 = center
        x, y = np.meshgrid(range(w), range(h))
        k = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
        norm = k.sum()
        return k / norm

    def clean_gausslist(self, kernelshape):
        for agauss in self.gausslist:
            if 'center' not in agauss:
                h, w = self.k_shape
                agauss['center'] = ((h - 1) / 2., (w - 1) / 2.)
            if 'modPolyDeg' not in agauss:
                agauss['modPolyDeg'] = 2
            if 'sx' not in agauss:
                agauss['sx'] = 2.
            if 'sy' not in agauss:
                agauss['sy'] = 2.

    def get_cmatrices(self):
        kh, kw = self.k_shape
        v, u = np.mgrid[:kh, :kw]
        c = []
        for aGauss in self.gausslist:
            n = aGauss['modPolyDeg'] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = self.gauss(center=aGauss['center'],
                                sx=aGauss['sx'], sy=aGauss['sy'])
            newc = [signal.convolve2d(self.refimage, gaussk * aU * aV,
                                      mode='same')
                    for i, aU in enumerate(allus)
                    for aV in allvs[:n - i]
                    ]
            c.extend(newc)
        return c

    def coeffstokernel(self, coeffs):
        kh, kw = self.k_shape
        v, u = np.mgrid[:kh, :kw]
        kernel = np.zeros((kh, kw))
        for aGauss in self.gausslist:
            n = aGauss['modPolyDeg'] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = self.gauss(center=aGauss['center'],
                                sx=aGauss['sx'], sy=aGauss['sy'])
            ind = 0
            for i, aU in enumerate(allus):
                for aV in allvs[:n - i]:
                    kernel += coeffs[ind] * aU * aV
                    ind += 1
            kernel *= gaussk
        return kernel

    def make_system(self):

        c = self.get_cmatrices()
        c_bkg = self.get_cmatrices_background()
        c.extend(c_bkg)

        if self.badpixmask is None:
            m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
            b = np.array([(self.image * ci).sum() for ci in c])
        else:
            # These next two lines take most of the computation time
            m = np.array([[(ci * cj)[~self.badpixmask].sum()
                         for ci in c] for cj in c])
            b = np.array([(self.image * ci)[~self.badpixmask].sum()
                         for ci in c])
        coeffs = np.linalg.solve(m, b)

        nkcoeffs = 0
        for aGauss in self.gausslist:
            n = aGauss['modPolyDeg'] + 1
            nkcoeffs += n * (n + 1) // 2

        self.kernel = self.coeffstokernel(coeffs[:nkcoeffs])
        self.background = self.coeffstobackground(coeffs[nkcoeffs:])
        opt_image = signal.convolve2d(
            self.refimage_data, self.kernel, mode='same') + self.background
        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(opt_image, mask=self.badpixmask)
        else:
            self.optimal_image = opt_image


class BramichStrategy(SubtractionStrategy):

    def __init__(self, image, refimage, kernelshape, bkg_degree, grid_shape):
        self.super.__init__(image, refimage, kernelshape, bkg_degree)
        # Deal here with the grid

    def get_cmatrices(self):
        kh, kw = self.k_shape
        h, w = self.refimage_data.shape
        c = []
        for i in range(kh):
            for j in range(kw):
                cij = np.zeros(self.refimage.shape)
                max_r = min(h, h - kh // 2 + i)
                min_r = max(0, i - kh // 2)
                max_c = min(w, w - kw // 2 + j)
                min_c = max(0, j - kw // 2)
                max_r_ref = min(h, h - i + kh // 2)
                min_r_ref = max(0, kh // 2 - i)
                max_c_ref = min(w, w - j + kw // 2)
                min_c_ref = max(0, kw // 2 - j)
                cij[min_r:max_r, min_c:max_c] = \
                    self.refimage[min_r_ref:max_r_ref, min_c_ref:max_c_ref]
                c.extend([cij])

        # This is more pythonic but much slower (50 times slower)
        # canonBasis = np.identity(kw*kh).reshape(kh*kw,kh,kw)
        # c.extend([signal.convolve2d(refimage, kij, mode='same')
        #                 for kij in canonBasis])
        # canonBasis = None

        return c

    def make_system(self):

        c = self.get_cmatrices()
        c_bkg = self.get_cmatrices_background()
        c.extend(c_bkg)

        if self.badpixmask is None:
            m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
            b = np.array([(self.image * ci).sum() for ci in c])
        else:
            # These next two lines take most of the computation time
            m = np.array([[(ci * cj)[~self.badpixmask].sum()
                         for ci in c] for cj in c])
            b = np.array([(self.image * ci)[~self.badpixmask].sum()
                         for ci in c])
        coeffs = np.linalg.solve(m, b)

        kh, kw = self.k_shape
        nkcoeffs = kh * kw
        self.kernel = coeffs[:nkcoeffs].reshape(self.k_shape)
        self.background = self.coeffstobackground(coeffs[nkcoeffs:])
        opt_image = signal.convolve2d(
            self.refimage_data, self.kernel, mode='same') + self.background
        if self.badpixmask is not None:
            self.optimal_image = np.ma.array(opt_image, mask=self.badpixmask)
        else:
            self.optimal_image = opt_image


class AdaptiveBramichStrategy(SubtractionStrategy):
    def __init__(self):



    def make_system(image, refimage, kernel_side, poly_degree, bkg_degree):
    import varconv

    # Check here for dimensions
    if image.ndim != 2:
        raise ValueError("Wrong dimensions for image")
    if refimage.ndim != 2:
        raise ValueError("Wrong dimensions for refimage")

    # Check here for types
    if image.dtype != np.float64:
        img64 = image.astype('float64')
    else:
        img64 = image
    if refimage.dtype != np.float64:
        ref64 = refimage.astype('float64')
    else:
        ref64 = refimage

    k_side = kernel_side
    k_shape = (k_side, k_side)
    img_data, ref_data, mask = separate_data_mask(img64, ref64, k_shape)

    c_bkg_degree = -1 if bkg_degree is None else bkg_degree
    poly_dof = (poly_degree + 1) * (poly_degree + 2) / 2
    m, b, conv = varconv.gen_matrix_system(img_data, ref_data,
                                           mask is not None, mask,
                                           k_side, poly_degree, c_bkg_degree)
    coeffs = np.linalg.solve(m, b)
    k_dof = k_side * k_side * poly_dof
    kernel = coeffs[:k_dof].reshape((k_side, k_side, poly_dof))
    opt_conv = varconv.convolve2d_adaptive(ref64, kernel, poly_degree)
    if bkg_degree is not None:
        background = _coeffstobackground(image.shape, coeffs[k_dof:])
        opt_image = opt_conv + background
    else:
        background = np.zeros(image.shape)
        opt_image = opt_conv

    if mask is not None:
        opt_image = np.ma.array(opt_image, mask=mask)

    return opt_image, kernel, background


def convolve2d_adaptive(image, kernel, poly_degree):
    import varconv

    # Check here for dimensions
    if image.ndim != 2:
        raise ValueError("Wrong dimensions for image")
    if kernel.ndim != 3:
        raise ValueError("Wrong dimensions for kernel")

    # Check here for types
    if image.dtype != np.float64:
        img64 = image.astype('float64')
    else:
        img64 = image
    if kernel.dtype != np.float64:
        k64 = kernel.astype('float64')
    else:
        k64 = kernel

    conv = varconv.convolve2d_adaptive(img64, k64, poly_degree)
    return conv


def optimal_system(image, refimage, method="AdaptiveBramich",
                   bkg_degree=3, *args, **kwargs):
    """kw for bramich: grid_shape
    kw for a-l: gausslist
    kw for adaptive: poly_degree"""

    DefaultStrategy = AdaptiveBramichStrategy # noqa
    all_strategies = {"AdaptiveBramich": AdaptiveBramichStrategy,
                      "Bramich": BramichStrategy,
                      "Alard-Lupton": AlardLuptonStrategy}
    DiffStrategy = all_strategies.get(method, DefaultStrategy) # noqa

    subt_strat = DiffStrategy(image, refimage, *args, **kwargs)
    opt_image = subt_strat.get_optimal_image()
    kernel = subt_strat.get_kernel()
    background = subt_strat.get_background()
    difference = subt_strat.get_difference()

    return difference, opt_image, kernel, background
