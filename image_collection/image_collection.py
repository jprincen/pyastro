""" Classes implmenting images collections. Image collection are just sets of images, and the 
    collection class allows operations on them
"""
from astropy.io import fits
import cv2
import numpy as np

from typing import List, Tuple


class ImageCollection(object):
    def __init__(self, images: List[np.ndarray]):
        """"""
        self._images = images

    def __getitem__(self, i) -> np.ndarray:
        return self._images[i]

    def __len__(self) -> int:
        return len(self._images)

    def mean(self) -> float:
        return np.mean(self._images, axis=0)

    def median(self) -> float:
        return np.median(self._images, axis=0)

    def __add__(self, operand):
        """Overload add so we can easily add images and image
        collections
        """
        if isinstance(operand, ImageCollection):
            # Check for length and size equivalence and add the
            # individual images
            raise NotImplemented("ImageCollection")
        elif isinstance(operand, np.ndarray):
            # Check for compatibility and add the array to each
            # element
            if operand.dtype != self._images[0].dtype:
                raise Exception("Incompatible dtype")
            if operand.shape != self._images[0].shape:
                raise Exception("Incompatible shapes")
            for i in range(0, len(self._images)):
                self._images[i] = self._images[i] + operand
        return self

    def __sub__(self, operand):
        """Support subtract"""
        if isinstance(operand, ImageCollection):
            # Check for length and size equivalence and subtract the
            # individual images
            raise NotImplemented("ImageCollection")
        elif isinstance(operand, np.ndarray):
            # Check for compatibility and subtract the array from each
            # element
            # TODO: might be able to use __add__ here as well
            if operand.dtype != self._images[0].dtype:
                raise Exception("Incompatible dtype")
            if operand.shape != self._images[0].shape:
                raise Exception("Incompatible shapes")
            for i in range(0, len(self._images)):
                self._images[i] = self._images[i] - operand
        return self

    def __mul__(self, operand):
        """Support multiply, including by a scalar"""
        if isinstance(operand, ImageCollection):
            raise NotImplemented("ImageCollection")
        elif isinstance(operand, np.ndarray):
            if operand.dtype != self._images[0].dtype:
                raise Exception("Incompatible dtype")
            if operand.shape != self._images[0].shape:
                raise Exception("Incompatible shapes")
            for i in range(0, len(self._images)):
                self._images[i] = np.multiply(self._images[i], operand)
        elif np.issctype(type(operand)):
            for i in range(0, len(self._images)):
                self._images[i] = self._images[i] * operand
        else:
            raise Exception("Incompatible type")
        return self

    def __truediv__(self, operand):
        """Support divide, including by a scalar"""
        if isinstance(operand, ImageCollection):
            raise NotImplemented("ImageCollection")
        elif isinstance(operand, np.ndarray):
            if operand.dtype != self._images[0].dtype:
                raise Exception("Incompatible dtype")
            if operand.shape != self._images[0].shape:
                raise Exception("Incompatible shapes")
            for i in range(0, len(self._images)):
                self._images[i] = np.divide(self._images[i], operand)
        elif np.issctype(type(operand)):
            for i in range(0, len(self._images)):
                self._images[i] = self._images[i] / operand
        else:
            raise Exception("Incompatible type")
        return self


class RawImageCollection(ImageCollection):
    def __init__(self, images: List[np.ndarray]):
        ImageCollection.__init__(self, images)

    @classmethod
    def from_files(cls, file_paths: List[str], max_value: int = 16384):
        """Init from a list of files"""
        images = []
        for fp in file_paths:
            hdul = fits.open(fp)
            images.append((hdul[0].data / max_value).astype(np.float32))
        return cls(images)


class RgbImageCollection(ImageCollection):
    def __init__(self, images: List[np.ndarray]):
        ImageCollection.__init__(self, images)

    @classmethod
    def from_rgb_files(cls, file_paths: List[str], max_value: int = 16384):
        images = []
        for fp in file_paths:
            img_bgr = np.clip(
                (cv2.imread(fp, cv2.IMREAD_ANYDEPTH) / max_value).astype(np.float32),
                0,
                1.0,
            )
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        return cls(images)

    @classmethod
    def from_raw_files(cls, file_paths: List[str], max_value: int = 16384):
        images = []
        for fp in file_paths:
            hdul = fits.open(fp)
            # Bayer conversion only works with 8 or 16 bit data
            image = cv2.cvtColor(hdul[0].data, cv2.COLOR_BAYER_BG2RGB)
            images.append(np.clip((image / max_value).astype(np.float32), 0, 1.0))
        return cls(images)

    @classmethod
    def from_raw_collection(
        cls,
        images: RawImageCollection,
        crop_r: Tuple[int, int] = None,
        crop_c: Tuple[int, int] = None,
        max_value: int = 16383,
    ):
        rgb_images = []
        # Note each image needs to first be converted to 16 bit from
        # float, the debayered, and converted back to float
        for img in images:
            img16 = np.clip((img * max_value).astype(np.uint16), 0, max_value)
            rgb16 = cv2.cvtColor(img16, cv2.COLOR_BAYER_BG2RGB)
            if crop_r is not None and crop_c is not None:
                rgb16 = rgb16[crop_r[0] : crop_r[1], crop_c[0] : crop_c[1]]
            rgb_images.append((rgb16 / max_value).astype(np.float32))
        return cls(rgb_images)
    
    def write(self, dir: str, base_name: str):
        """ Write out images as uncompressed PNG """
        import os
        for i, img in enumerate(self._images):
            file_name = os.path.join(dir, base_name + '_' + '{0:03}'.format(i) + '.png')
            i16 = (img*(2**16-1)).astype(np.uint16)
            bgr = cv2.cvtColor(i16, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_name, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

