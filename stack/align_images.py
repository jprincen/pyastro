import astroalign as aa
from tqdm.auto import tqdm

from image_collection import RgbImageCollection


def align_images(
    images: RgbImageCollection, ref_image_index: int = 0
) -> RgbImageCollection:
    """Align images to a reference image, and return the aligned image collection"""
    a_images = []
    for img in tqdm(images):
        reg_img, _ = aa.register(img, images[ref_image_index])
        a_images.append(reg_img)

    return RgbImageCollection(a_images)
