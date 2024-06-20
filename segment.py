import requests
import json
import os
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from base64 import b64decode, b64encode
from ada_pallete import ada_palette
from collections import Counter


palette = [tuple(i) for i in ada_palette.tolist()]
id2label =  {
    "0": "wall",
    "1": "building",
    "2": "sky",
    "3": "floor",
    "4": "tree",
    "5": "ceiling",
    "6": "road",
    "7": "bed ",
    "8": "windowpane",
    "9": "grass",
    "10": "cabinet",
    "11": "sidewalk",
    "12": "person",
    "13": "earth",
    "14": "door",
    "15": "table",
    "16": "mountain",
    "17": "plant",
    "18": "curtain",
    "19": "chair",
    "20": "car",
    "21": "water",
    "22": "painting",
    "23": "sofa",
    "24": "shelf",
    "25": "house",
    "26": "sea",
    "27": "mirror",
    "28": "rug",
    "29": "field",
    "30": "armchair",
    "31": "seat",
    "32": "fence",
    "33": "desk",
    "34": "rock",
    "35": "wardrobe",
    "36": "lamp",
    "37": "bathtub",
    "38": "railing",
    "39": "cushion",
    "40": "base",
    "41": "box",
    "42": "column",
    "43": "signboard",
    "44": "chest of drawers",
    "45": "counter",
    "46": "sand",
    "47": "sink",
    "48": "skyscraper",
    "49": "fireplace",
    "50": "refrigerator",
    "51": "grandstand",
    "52": "path",
    "53": "stairs",
    "54": "runway",
    "55": "case",
    "56": "pool table",
    "57": "pillow",
    "58": "screen door",
    "59": "stairway",
    "60": "river",
    "61": "bridge",
    "62": "bookcase",
    "63": "blind",
    "64": "coffee table",
    "65": "toilet",
    "66": "flower",
    "67": "book",
    "68": "hill",
    "69": "bench",
    "70": "countertop",
    "71": "stove",
    "72": "palm",
    "73": "kitchen island",
    "74": "computer",
    "75": "swivel chair",
    "76": "boat",
    "77": "bar",
    "78": "arcade machine",
    "79": "hovel",
    "80": "bus",
    "81": "towel",
    "82": "light",
    "83": "truck",
    "84": "tower",
    "85": "chandelier",
    "86": "awning",
    "87": "streetlight",
    "88": "booth",
    "89": "television receiver",
    "90": "airplane",
    "91": "dirt track",
    "92": "apparel",
    "93": "pole",
    "94": "land",
    "95": "bannister",
    "96": "escalator",
    "97": "ottoman",
    "98": "bottle",
    "99": "buffet",
    "100": "poster",
    "101": "stage",
    "102": "van",
    "103": "ship",
    "104": "fountain",
    "105": "conveyer belt",
    "106": "canopy",
    "107": "washer",
    "108": "plaything",
    "109": "swimming pool",
    "110": "stool",
    "111": "barrel",
    "112": "basket",
    "113": "waterfall",
    "114": "tent",
    "115": "bag",
    "116": "minibike",
    "117": "cradle",
    "118": "oven",
    "119": "ball",
    "120": "food",
    "121": "step",
    "122": "tank",
    "123": "trade name",
    "124": "microwave",
    "125": "pot",
    "126": "animal",
    "127": "bicycle",
    "128": "lake",
    "129": "dishwasher",
    "130": "screen",
    "131": "blanket",
    "132": "sculpture",
    "133": "hood",
    "134": "sconce",
    "135": "vase",
    "136": "traffic light",
    "137": "tray",
    "138": "ashcan",
    "139": "fan",
    "140": "pier",
    "141": "crt screen",
    "142": "plate",
    "143": "monitor",
    "144": "bulletin board",
    "145": "shower",
    "146": "radiator",
    "147": "glass",
    "148": "clock",
    "149": "flag"
  }

floor_items = [
    "floor", "road", "bed", "grass", "sidewalk", "table", "plant", "chair", "sofa", "armchair", "seat", "desk", 
    "rug", "stairs", "runway", "pool table", "bench", "countertop", "stove", "kitchen island", "boat", "hovel", 
    "bus", "truck", "ottoman", "conveyer belt", "plaything", "stool", "barrel", "basket", "minibike", "cradle", 
    "bicycle", "blanket", "tank", "trade name", "bag", "step", "food", "plate"
]

wall_items = [
    "wall", "windowpane", "cabinet", "door", "painting", "shelf", "mirror", "wardrobe", "column", "signboard", 
    "chest of drawers", "counter", "sink", "refrigerator", "fireplace", "path", "case", "pillow", "screen door", 
    "blind", "computer", "television receiver", "apparel", "pole", "bannister", "poster", "stage", "washer", 
    "plaything", "microwave", "pot", "animal", "dishwasher", "screen", "sculpture", "hood", "vase", "bulletin board", 
    "radiator", "glass", "clock", "flag","bookcase", "curtain"
]

ceiling_items = [
    "ceiling", "lamp", "chandelier", "light", "awning", "streetlight", "fan", "traffic light", "sconce"
]
label2id = {id2label[key]:key for key in id2label.keys()}

def get_integers(matrix):
    """
    This function takes a numpy matrix of integers and returns a list of integers
    that appear exactly once or exactly twice in the matrix.
    
    Parameters:
    matrix (numpy.ndarray): A numpy matrix of integers.
    
    Returns:
    list: A list of integers that appear exactly once or exactly twice.
    """
    # Flatten the matrix to a 1D array
    flattened_array = matrix.flatten()
    
    # Count the occurrences of each integer in the flattened array
    counter = Counter(flattened_array)
    
    # Extract integers that appear exactly once or exactly twice
    result = [num for num, count in counter.items() if count >= 1]
    
    return result


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert an image to base64 string.

    Args:
        image (Image.Image): Image to convert.
        format (str, optional): Image format. Defaults to "PNG".

    Returns:
        str: Base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    return b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert a base64 string to an image.

    Args:
        base64_str (str): Base64 string.

    Returns:
        Image.Image: Image.
    """
    image_data = b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def get_ade20k_semantic_map(image) -> Image:
    ade_semantic_map_api_url = "http://192.168.32.42:8035/ade-semantic-map"
    data = {
        "image_base64": image_to_base64(image),
        "rgb_semantic_map": False,
        "resolution_mode": "medium",
    }
    # response = requests.post(ade_semantic_map_api_url, json=data)
    # response.raise_for_status()
    # buffer = BytesIO(response.content)
    # result = Image.open(buffer)
    # segmap = np.asarray(result)
    # rgbs = [tuple(i) for i in segmap.reshape(-1, 3).tolist()]
    # indices = np.array([palette.index(i) for i in rgbs], dtype=np.uint8)
    # indices = indices.reshape(segmap.shape[:2])
    # id_labels = get_integers(indices)
    # labels = [id2label[str(i)] for i in id_labels]
    # data_image = {}

    # for label in labels:

    #     mask = np.zeros_like(indices, dtype=np.uint8)
    #     mask_condition = indices==int(label2id[label])
    #     mask[mask_condition]=255
    #     data_image[label] = mask
    response = requests.post(ade_semantic_map_api_url, json=data)
    response.raise_for_status()
    buffer = BytesIO(response.content)
    buffer.seek(0)
    result = Image.open(buffer)
    segmap = np.asarray(result)
    data_image = {}
    for idx in np.unique(segmap):
        l = id2label[str(idx)]
        m = segmap == idx
        data_image[l] = m
    return data_image


def calculate_intersection_over_image_area(mask1, mask2):
        """
        Calculate the intersection of two masks over the whole image area.
        
        Parameters:
        mask1 (numpy.ndarray): The first mask, of shape (768, 1024).
        mask2 (numpy.ndarray): The second mask, of shape (768, 1024).
        
        Returns:
        float: The intersection over the whole image area.
        """
        # Ensure the masks are boolean arrays
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        
        # Calculate the intersection
        intersection = np.logical_and(mask1, mask2)
        
        # Calculate the intersection over the whole image area
        intersection_over_image_area = np.sum(intersection) / intersection.size
        
        return intersection_over_image_area


# this def is used to calculate a mask intersection with the 
# masks of different objectsd in the image then determin if it is an object placed on a wall, ceiling or floor
def Location_detector(image, mask):
    data = get_ade20k_semantic_map(image=image)
    intersection_dict = {}
    for labele in data.keys():
        intersection  = calculate_intersection_over_image_area(mask, data[labele])
        if intersection != 0:
            intersection_dict[labele] = intersection

    intersection_dict = dict(sorted(intersection_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    locations = []
    for object in intersection_dict.keys():
        if object in floor_items:
            locations.append('floor')
        elif object in ceiling_items:
            locations.append('ceiling')
        elif object in wall_items:
            locations.append('wall')

    if 'floor' in locations:
        return 'floor'
    elif 'ceiling' in locations:
        return 'ceiling'
    else :  
        return 'wall'
    