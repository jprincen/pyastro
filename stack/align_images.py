import astroalign as aa
from tqdm.auto import tqdm

from image_collection import RgbImageCollection


def align_images(
    images: RgbImageCollection, ref_image_index: int = 0
) -> RgbImageCollection:
    """Align images to a reference image, and return the aligned image collection"""
    a_images = []
    for i, img in tqdm(zip(range(len(images)), images), total=len(images)):
        try: 
            reg_img, _ = aa.register(img, images[ref_image_index])
            a_images.append(reg_img)
        except Exception as err:
            print(f"Error {type(err)} in registration for image {i}")
    return RgbImageCollection(a_images)
