import numpy as np


class Drizzle(object):
    """A simplified version of drizzle super-resolution, which uses axis aligned
    square cell overlap to interpolate, rather than oriented squares
    """

    def __init__(self, rows: int, cols: int, chans: int, zoom: float = 2.0):
        self._zoom = zoom
        self._chans = chans
        # Add padding for the border, so we can clip results
        self._image = np.zeros((rows + 4, cols + 4, chans))

    def add_image(self, tx: np.ndarray, img: np.ndarray, weight: float = 1.0):
        """Add an image to the output with drizzle. tx is a 3x3 similarity transform,
        image can be rgb or gray scale
        """
        # Create the input index array for rows and columns
        # Note that the 'x' coordinate is stored in element 0, and 'y' coordinate is stored in element 1
        in_index = np.array(
            [
                np.array([c, r, 1])
                for r in range(img.shape[0])
                for c in range(img.shape[1])
            ]
        ).reshape((img.shape[0], img.shape[1], 3))
        # Indices of the input image mapped onto output. These are floating point
        # Note we offset the index by 2.0 so we can use the border for clipping
        # The center part of the image is returned so border is removed in the end
        tx_index = self._zoom * np.matmul(in_index, tx.transpose()) + 2.0
        # Take the integer part which will give the index into the cells, separated
        # by the zoom factor
        int_tx_index = np.floor(tx_index).astype(int)
        # Clip to boundary.
        # Note this means boundary pixels are not correct, so ignore them in the end
        int_tx_index[:, :, 0] = np.clip(
            int_tx_index[:, :, 0], 0, self._image.shape[1] - 2
        )
        int_tx_index[:, :, 1] = np.clip(
            int_tx_index[:, :, 1], 0, self._image.shape[0] - 2
        )
        # Use fraction to determine weights for surrounding pixels
        frac_tx_index = tx_index - int_tx_index
        # Now need to calculate 4 weight matrices: (1-dx).(1-dy), dx.(1-dy), (1-dx).dy, dx.dy
        w00 = np.expand_dims(
            (1.0 - frac_tx_index[:, :, 0]) * (1.0 - frac_tx_index[:, :, 1]), axis=2
        )  # (1-dx)(1-dy)
        w01 = np.expand_dims(
            frac_tx_index[:, :, 0] * (1.0 - frac_tx_index[:, :, 1]), axis=2
        )  # dx.(1-dy)
        w10 = np.expand_dims(
            (1.0 - frac_tx_index[:, :, 0]) * frac_tx_index[:, :, 1], axis=2
        )  # (1-dx).dy
        w11 = np.expand_dims(
            frac_tx_index[:, :, 0] * frac_tx_index[:, :, 1], axis=2
        )  # dx.dy
        # Now add up the contributions into the output
        self._image[int_tx_index[:, :, 1], int_tx_index[:, :, 0], :] = (
            self._image[int_tx_index[:, :, 1], int_tx_index[:, :, 0], :]
            + w00 * img * weight
        )
        self._image[int_tx_index[:, :, 1] + 1, int_tx_index[:, :, 0], :] = (
            self._image[int_tx_index[:, :, 1] + 1, int_tx_index[:, :, 0], :]
            + w10 * img * weight
        )
        self._image[int_tx_index[:, :, 1], int_tx_index[:, :, 0] + 1, :] = (
            self._image[int_tx_index[:, :, 1], int_tx_index[:, :, 0] + 1, :]
            + w01 * img * weight
        )
        self._image[int_tx_index[:, :, 1] + 1, int_tx_index[:, :, 0] + 1, :] = (
            self._image[int_tx_index[:, :, 1] + 1, int_tx_index[:, :, 0] + 1, :]
            + w11 * img * weight
        )

    @property
    def image(self):
        # Return the valid center part of the image
        return self._image[
            2 : self._image.shape[0] - 2, 2 : self._image.shape[1] - 2, :
        ]
