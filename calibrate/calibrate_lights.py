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
    # Scaling value, selected to ensure mean value of calibrated light frame
    # doesn't change after calibration
    s0 = np.average(master_flat_c)
    # Multiplicative factor
    mxy = master_flat_c / s0
    # Now for each frame Ireal = (Idetected - master_dark_uc) / mxy
    # See reference
    lights_c = lights - master_dark_uc
    lights_c = lights_c / mxy
    return lights_c
