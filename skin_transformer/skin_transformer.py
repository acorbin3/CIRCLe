import numpy as np
from derm_ita import get_ita, get_kinyanjui_type, get_cropped_center_ita, get_kinyanjui_groh_type
import io
from skin_transformer.fitzpatrick_utilities import Fitzpatrick_Skin_Type, random_FST, random_ita_from_FST
from PIL import ImageOps, Image
import cv2


def safe_convert(x, new_dtype):
    info = np.iinfo(new_dtype)
    return x.clip(info.min, info.max).astype(new_dtype)


def transform_image(image, mask, desired_fst=None, image_ita=None, verbose=False):
    """
    1. Compute ITA of current image and retrieve Fitzpatrick skin type
    2. Select random FST that’s different
    3. Select random ITA number within range of selected FST
    4. Compute difference  original ITA – new desired ITA
    5. Adjusted b = old b value + (difference * .5)
    6. Adjusted L = old L value + (difference * .12)
    :image - rgb image
    """

    if verbose: print(f"image type {type(image)}")
    if image_ita is None:
        image_ita = get_cropped_center_ita(image)
    if verbose: print(f"ITA {image_ita}")

    fst = Fitzpatrick_Skin_Type[f"_{get_kinyanjui_groh_type(image_ita)}"]
    if verbose: print(f"FST {fst}")

    # select random FST unless specified
    if desired_fst is None:
        random_fst = random_FST(exclude=fst)
    else:
        random_fst = desired_fst

    if verbose: print(f"Random FST: {random_fst}")

    random_ita = random_ita_from_FST(random_fst)
    if verbose: print(f"Random ita: {random_ita}")

    difference = image_ita - random_ita
    if verbose: print(f"Difference ita: {difference}")

    # convert to boolean array
    mask = ImageOps.invert(mask)
    mask = mask.convert("1")

    # convert base image to NumPy array
    image_array = np.array(image)

    # Convert the base image NumPy arrays to a LAB color space
    if image_array.shape[2] == 3:
        lab_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
        if verbose:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            print('The image is in the BGR color space')
            print(f"image_array: {image_array.dtype}")
            cv2.imwrite("test_02_before_lab.jpg", image_array)
    else:
        lab_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        if verbose:
            print('The image is in the BGR color space')
            print(f"image_array: {image_array.dtype}")
            cv2.imwrite("test_02_before_lab.jpg", image_array)

    if verbose:
        cv2.imwrite("test_03_after_lab.jpg", lab_image)

    # Create the modifiers
    if random_fst.value < fst.value:
        b_modifier = .31
        l_modifier = -.09
        a_modifier = -.06
    else:
        if difference > 25:
            b_modifier = .76
            l_modifier = -.85
        else:
            b_modifier = .56
            l_modifier = -.65
        a_modifier = .1
    if verbose: print(
        f"b_modifier: {b_modifier} random_fst.value {random_fst} {random_fst.value} fst.value {fst} {fst.value}")

    # Convert to int64's incase we have negative numbers due to the modifier
    lab_image = safe_convert(lab_image, np.int64)

    lab_image[:, :, 2][mask] += int(difference * b_modifier)
    lab_image[:, :, 1][mask] += int(difference * a_modifier)
    lab_image[:, :, 0][mask] += int(difference * l_modifier)

    # convert back to uint8's as thats what cv2 expects
    lab_image = safe_convert(lab_image, np.uint8)

    # Convert the image back to the original color space
    adjusted_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    if verbose:
        # Save the LAB image to a file
        cv2.imwrite("test.jpg", adjusted_image)

        image = Image.open(io.BytesIO(open("test.jpg", "rb").read()))

        image_ita = get_cropped_center_ita(image)
        print(f"Updated ITA {image_ita}")

    # To get a PIL image type you need to do this:
    # Image.fromarray(util.img_as_ubyte(rgb))
    # source: https://stackoverflow.com/a/55893334
    return adjusted_image
