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

__version__ = '0.1a3dev'

import numpy as np
from scipy import signal
from scipy import ndimage


def _gauss(shape, center, sx, sy):
    h, w = shape
    x0, y0 = center
    x, y = np.meshgrid(range(w), range(h))
    k = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
    norm = k.sum()
    return k / norm


def _cforgauss(refimage, kernelshape, gausslist):
    kh, kw = kernelshape

    v, u = np.mgrid[:kh, :kw]
    c = []
    for aGauss in gausslist:
        n = aGauss['modPolyDeg'] + 1
        allus = [pow(u, i) for i in range(n)]
        allvs = [pow(v, i) for i in range(n)]
        gaussk = _gauss(shape=(kh, kw), center=aGauss['center'],
                        sx=aGauss['sx'], sy=aGauss['sy'])
        newc = [signal.convolve2d(refimage, gaussk * aU * aV, mode='same')
                for i, aU in enumerate(allus)
                for aV in allvs[:n - i]
                ]
        c.extend(newc)
    return c


def _cfordelta(refimage, kernelshape):
    kh, kw = kernelshape
    h, w = refimage.shape
    c = []
    for i in range(kh):
        for j in range(kw):
            cij = np.zeros(refimage.shape)
            max_row = min(h, h - kh // 2 + i)
            min_row = max(0, i - kh // 2)
            max_col = min(w, w - kw // 2 + j)
            min_col = max(0, j - kw // 2)
            max_row_ref = min(h, h - i + kh // 2)
            min_row_ref = max(0, kh // 2 - i)
            max_col_ref = min(w, w - j + kw // 2)
            min_col_ref = max(0, kw // 2 - j)
            cij[min_row:max_row, min_col:max_col] = \
                refimage[min_row_ref:max_row_ref, min_col_ref:max_col_ref]
            c.extend([cij])

    # This is more pythonic but much slower (50 times slower)
    # canonBasis = np.identity(kw*kh).reshape(kh*kw,kh,kw)
    # c.extend([signal.convolve2d(refimage, kij, mode='same')
    #                 for kij in canonBasis])
    # canonBasis = None

    return c


def _getcvectors(refimage, kernelshape, gausslist, modbkgdeg):
    c = []
    if gausslist is not None:
        c.extend(_cforgauss(refimage, kernelshape, gausslist))
    else:
        c.extend(_cfordelta(refimage, kernelshape))

    # finally add here the background variation coefficients:
    h, w = refimage.shape
    y, x = np.mgrid[:h, :w]
    allxs = [pow(x, i) for i in range(modbkgdeg + 1)]
    allys = [pow(y, i) for i in range(modbkgdeg + 1)]

    newc = [anX * aY for i, anX in enumerate(allxs)
            for aY in allys[:modbkgdeg + 1 - i]]

    c.extend(newc)
    return c


def _coeffstokernel(coeffs, gausslist, kernelshape):
    kh, kw = kernelshape
    if gausslist is None:
        kernel = coeffs[:kw * kh].reshape(kh, kw)
    else:
        v, u = np.mgrid[:kh, :kw]
        kernel = np.zeros((kh, kw))
        for aGauss in gausslist:
            n = aGauss['modPolyDeg'] + 1
            allus = [pow(u, i) for i in range(n)]
            allvs = [pow(v, i) for i in range(n)]
            gaussk = _gauss(shape=kernelshape, center=aGauss['center'],
                            sx=aGauss['sx'], sy=aGauss['sy'])
            ind = 0
            for i, aU in enumerate(allus):
                for aV in allvs[:n - i]:
                    kernel += coeffs[ind] * aU * aV
                    ind += 1
            kernel *= gaussk
    return kernel


def _coeffstobackground(shape, coeffs, bkgdeg=None):
    if bkgdeg is None:
        bkgdeg = int(-1.5 + 0.5 * np.sqrt(9 + 8 * (len(coeffs) - 1)))
    h, w = shape
    y, x = np.mgrid[:h, :w]
    allxs = [pow(x, i) for i in range(bkgdeg + 1)]
    allys = [pow(y, i) for i in range(bkgdeg + 1)]
    mybkg = np.zeros(shape)
    ind = 0
    for i, anX in enumerate(allxs):
        for aY in allys[:bkgdeg + 1 - i]:
            mybkg += coeffs[ind] * anX * aY
            ind += 1
    return mybkg


def clean_gausslist(gausslist, kernelshape):
    if gausslist is None:
        return
    for agauss in gausslist:
        if 'center' not in agauss:
            h, w = kernelshape
            agauss['center'] = ((h - 1) / 2., (w - 1) / 2.)
        if 'modPolyDeg' not in agauss:
            agauss['modPolyDeg'] = 2
        if 'sx' not in agauss:
            agauss['sx'] = 2.
        if 'sy' not in agauss:
            agauss['sy'] = 2.


def optimalkernelandbkg(image, refimage, gausslist=None,
                        bkgdegree=3, kernelshape=(11, 11)):
    """Do Optimal Image Subtraction and return optimal kernel and background.

    This is an implementation of the Optimal Image Subtraction algorith of
    Alard&Lupton. It returns the best kernel and background fit that match the
    two images.

    gausslist is a list of dictionaries containing data of the gaussians
    used in the decomposition of the kernel. Dictionary keywords are:
    center, sx, sy, modPolyDeg
    If gausslist is None (default value), the OIS will try to optimize
    the value of each pixel in the kernel.

    bkgdegree is the degree of the polynomial to fit the background.

    kernelshape is the shape of the kernel to use.

    Return (optimal_image, kernel, background)
    """

    kh, kw = kernelshape
    if kw % 2 != 1 or kh % 2 != 1:
        print("This can only work with kernels of odd sizes.")
        return None, None, None

    clean_gausslist(gausslist, kernelshape)

    def has_mask(image):
        is_masked_array = isinstance(image, np.ma.MaskedArray)
        if is_masked_array and isinstance(image.mask, np.ndarray):
            return True
        return False

    badpixmask = None
    if has_mask(refimage):
        badpixmask = ndimage.binary_dilation(
            refimage.mask.astype('uint8'), structure=np.ones(kernelshape))\
            .astype('bool')
        if has_mask(image):
            badpixmask += image.mask
    elif has_mask(image):
        badpixmask = image.mask

    c = _getcvectors(refimage, kernelshape, gausslist, bkgdegree)

    if badpixmask is None:
        m = np.array([[(ci * cj).sum() for ci in c] for cj in c])
        b = np.array([(image * ci).sum() for ci in c])
    else:
        # These next two lines take most of the computation time
        m = np.array([[(ci * cj)[~badpixmask].sum() for ci in c] for cj in c])
        b = np.array([(image * ci)[~badpixmask].sum() for ci in c])

    coeffs = np.linalg.solve(m, b)

    # nkcoeffs is the number of coefficients related to the kernel fit,
    # not the background fit
    if gausslist is None:
        nkcoeffs = kh * kw
    else:
        nkcoeffs = 0
        for aGauss in gausslist:
            n = aGauss['modPolyDeg'] + 1
            nkcoeffs += n * (n + 1) // 2

    kernel = _coeffstokernel(coeffs[:nkcoeffs], gausslist, kernelshape)
    background = _coeffstobackground(image.shape, coeffs[nkcoeffs:])
    if has_mask(refimage) or has_mask(image):
        background = np.ma.array(background, mask=badpixmask)
    optimal_image = signal.convolve2d(refimage, kernel, mode='same') \
        + background

    return optimal_image, kernel, background


def subtractongrid(image, refimage, gausslist=None, bkgdegree=3,
                   kernelshape=(11, 11), gridshape=(2, 2)):
    """Implement Optimal Image Subtraction on a grid and return the optimal
    subtraction

    This is an implementation of the Optimal Image Subtraction algorith of
    Alard&Lupton(1998) and Bramich(2010)
    It returns the optimal subtraction between image and refimage.

    gausslist is a list of dictionaries containing data of the gaussians
    used in the multigaussian decomposition of the kernel [Alard&Lupton, 1998].
    Dictionary keywords are:
    center, sx, sy, modPolyDeg

    If gausslist is None (default value), the OIS will try to optimize the
    value of each pixel in the kernel [Bramich, 2010].

    bkgdegree is the degree of the polynomial to fit the background.

    kernelshape is the shape of the kernel to use.

    gridshape is a tuple containing the number of vertical and horizontal
    divisions of the grid.

    This method does not interpolate between the grids.

    Return (subtraction_array)
    """
    ny, nx = gridshape
    h, w = image.shape
    kh, kw = kernelshape

    # normal slices with no border
    stamps_y = [slice(h * i / ny, h * (i + 1) / ny, None) for i in range(ny)]
    stamps_x = [slice(w * i / nx, w * (i + 1) / nx, None) for i in range(nx)]

    # slices with borders where possible
    slc_wborder_y = [slice(max(0, h * i / ny - (kh - 1) / 2),
                           min(h, h * (i + 1) / ny + (kh - 1) / 2), None)
                     for i in range(ny)]
    slc_wborder_x = [slice(max(0, w * i / nx - (kw - 1) / 2),
                           min(w, w * (i + 1) / nx + (kw - 1) / 2), None)
                     for i in range(nx)]

    img_stamps = [image[sly, slx] for sly in slc_wborder_y
                  for slx in slc_wborder_x]
    ref_stamps = [refimage[sly, slx] for sly in slc_wborder_y
                  for slx in slc_wborder_x]

    # After we do the subtraction we need to crop the extra borders in the
    # stamps.
    # The recover_slices are the prescription for what to crop on each stamp.
    recover_slices = []
    for i in range(ny):
        start_border_y = slc_wborder_y[i].start
        stop_border_y = slc_wborder_y[i].stop
        sly_stop = h * (i + 1) / ny - stop_border_y
        if sly_stop == 0:
            sly_stop = None
        sly = slice(h * i / ny - start_border_y, sly_stop, None)
        for j in range(nx):
            start_border_x = slc_wborder_x[j].start
            stop_border_x = slc_wborder_x[j].stop
            slx_stop = w * (j + 1) / nx - stop_border_x
            if slx_stop == 0:
                slx_stop = None
            slx = slice(w * j / nx - start_border_x, slx_stop, None)
            recover_slices.append([sly, slx])

    # Here do the subtraction on each stamp
    subtract_collage = np.ma.empty(image.shape)
    stamp_slices = [[asly, aslx] for asly in stamps_y for aslx in stamps_x]
    for ind, ((sly_out, slx_out), (sly_in, slx_in)) in \
            enumerate(zip(recover_slices, stamp_slices)):
        opti, ki, bgi = optimalkernelandbkg(img_stamps[ind],
                                            ref_stamps[ind],
                                            gausslist,
                                            bkgdegree,
                                            kernelshape)

        subtract_collage[sly_in, slx_in] = \
            (img_stamps[ind] - opti)[sly_out, slx_out]

    return subtract_collage


def find_best_variable_kernel(image, refimage, kernel_side, poly_degree):
    import varconv

    image = np.random.random((10, 10))
    refimage = np.random.random((10, 10))
    m, b, conv = varconv.gen_matrix_system(image, refimage, kernel_side,
                                           poly_degree)
    # coeffs = np.linalg.solve(m, b)
    print("Shape of M", m.shape)
    print("Shape of b", b.shape)
    print("Shape of conv", conv.shape)
    # print("Coeffs: ", coeffs)
    return


if __name__ == '__main__':
    print(__doc__)
    print("\nVersion: {}".format(__version__))
