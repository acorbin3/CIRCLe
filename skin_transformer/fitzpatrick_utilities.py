from enum import Enum

from numpy.random.mtrand import randint
import random

Fitzpatrick_Skin_Type = Enum("Fitzpatrick_Skin_Type", ["_1", "_2", "_3", "_4", "_5", "_6"])


def random_ita_from_FST(skin_type: Fitzpatrick_Skin_Type):
    """
  This function will return a random number between the ITA ranges provided in
  https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf
    For skin types of 6 and 1 there are open ranges. The selected end ranges comes from computing the ITA
    on all the ISIC 2017 dataset images and takeing the largest ranges.
  :skin_type - input of the Fitzpatrick skin type
  """
    if skin_type == Fitzpatrick_Skin_Type._6:
        return randint(-10, 10)
    elif skin_type == Fitzpatrick_Skin_Type._5:
        return randint(10, 19)
    elif skin_type == Fitzpatrick_Skin_Type._4:
        return randint(19, 28)
    elif skin_type == Fitzpatrick_Skin_Type._3:
        return randint(28, 41)
    elif skin_type == Fitzpatrick_Skin_Type._2:
        return randint(41, 55)
    elif skin_type == Fitzpatrick_Skin_Type._1:
        return randint(55, 90)


def random_FST(exclude=None):
  """
  This function will randomly pick a fitzpatrick skin type
  :exclude - given a FST, this will not be chosen when randomly picking a FST
  """
  if exclude is None:
    return Fitzpatrick_Skin_Type[f"_{random.randint(1, 6)}"]
  else:
    while True:
      fst = Fitzpatrick_Skin_Type[f"_{random.randint(1, 6)}"]
      if fst != exclude:
        return fst