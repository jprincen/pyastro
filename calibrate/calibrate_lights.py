from image_collection import RawImageCollection
import numpy as np


def calibrate_lights(
    biases: RawImageCollection,
    darks: RawImageCollection,
    flats: RawImageCollection,
    lights: RawImageCollection,
) -> RawImageCollection:
    """Given all the calibration frames calculate a set of calibrate light images and
    return as a RawImageCollection.
    Reference: https://pixinsight.com/doc/tools/ImageCalibration/ImageCalibration.html
    NOTE: This is without dark frame optimization for now!
    """
    # Master Bias
    master_bias = biases.mean()
    # Uncalibrated Master Dark
    master_dark_uc = darks.mean()
    # Calibrated Master Flat
    master_flat_c = flats.mean() - master_bias
    # Select different scaling value for each CFA element, so we don't alter
    # the color balance of the result
    f_red = master_flat_c[::2, ::2]
    f_blue = master_flat_c[1::2, 1::2]
    f_green_1 = master_flat_c[1::2, ::2]
    f_green_2 = master_flat_c[::2, 1::2]
    # Form master flat scaling
    # TODO:::Should probably just use center of image, so
    # heavy vignetting doesn't lead to under estimation of averages
    mxy = np.zeros(master_flat_c.shape).astype(np.float32)
    mxy[::2, ::2] = f_red / np.average(f_red)
    mxy[1::2, 1::2] = f_blue / np.average(f_blue)
    mxy[1::2, ::2] = f_green_1 / np.average(f_green_1)
    mxy[::2, 1::2] = f_green_2 / np.average(f_green_2)

    # Now for each frame Ireal = (Idetected - master_dark_uc) / mxy
    # See reference
    lights_c = lights - master_dark_uc
    lights_c = lights_c / mxy
    return lights_c
