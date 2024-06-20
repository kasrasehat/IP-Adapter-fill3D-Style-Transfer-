from controlnet_aux import NormalBaeDetector
from PIL import Image
import numpy as np
from xyzrotation import rotate_image
import cv2

from PIL import Image


def replace_non_masked_part(init_image, mask, reference_image):
    """
    Replace the non-masked part of the initial image with parts from the reference image without resizing the reference image.

    Parameters:
    init_image (PIL.Image): The initial image object.
    mask (PIL.Image): The mask image object (single-channel).
    reference_image (PIL.Image): The reference image object.

    Returns:
    PIL.Image: The modified initial image with the non-masked areas replaced by the reference image.
    """
    # Ensure mask is in grayscale
    mask = mask.convert('L')

    # Resize reference to match init_image's dimensions if necessary
    if reference_image.size != init_image.size:
        reference_image = reference_image.resize(init_image.size, Image.Resampling.LANCZOS)
    # # Crop the reference image if it's larger than the initial image
    # if reference_image.size != init_image.size:
    #     reference_image = reference_image.crop((0, 0, init_image.width, init_image.height))

    # Invert the mask to replace non-masked areas
    inverted_mask = Image.eval(mask, lambda x: 255 if x == 0 else 0)

    # Composite the images using the inverted mask
    # Where the inverted mask is white (255), pixels from the reference image are used.
    result_image = Image.composite(reference_image, init_image, mask)

    return result_image


def convert_cv2_to_pil(cv2_image):
    """
    Convert an OpenCV image (numpy array) in BGR format to a PIL Image in RGB format.

    Parameters:
    cv2_image (numpy.ndarray): The image in OpenCV BGR format.

    Returns:
    PIL.Image: The converted image in RGB format.
    """
    # Convert from BGR to RGB
    # rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert from numpy array to PIL Image
    pil_image = Image.fromarray(cv2_image)

    return pil_image


def calculate_masked_normal_vector_pil(normal_map, mask):
    """
    Calculates the normalized average normal vector from the masked region of a normal map using PIL.

    Parameters:
    normal_map_path (str): Path to the normal map image file.
    mask_path (str): Path to the mask image file.

    Returns:
    np.ndarray: A normalized 3D vector representing the average normal in the masked region.
    """

    # Convert images to numpy arrays
    normal_map_np = np.array(normal_map)
    mask_np = np.array(mask)

    # Ensure mask is a boolean array (assuming the mask is grayscale)
    mask_np = mask_np > 127  # Adjust threshold as necessary

    # Apply the mask to each channel of the normal map
    masked_normal_map = normal_map_np * mask_np[:, :, np.newaxis]  # Expand mask dimensions

    # Compute the mean normal vector in the masked region
    mean_normal = np.array([
        np.mean(masked_normal_map[:, :, 0]),
        np.mean(masked_normal_map[:, :, 1]),
        np.mean(masked_normal_map[:, :, 2])
    ])

    # Normalize the mean normal vector
    norm = np.linalg.norm(mean_normal)
    if norm == 0:
        return mean_normal  # Return zero vector if norm is zero to avoid division by zero
    normalized_mean_normal = mean_normal / norm

    return normalized_mean_normal


def normal_image(image: Image.Image):
    processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(image)
    return control_image


def vector_to_rotation_matrix(a, b):
    # Normalize vectors
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Cross and dot product
    v = np.cross(a, b)
    c = np.dot(a, b)

    # Skew-symmetric cross-product matrix
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    # Rotation matrix
    R = np.eye(3) + vx + np.matmul(vx, vx) * (1 / (1 + c))

    return R


def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    final_rotation = np.degrees(np.array([x, y, z]))
    return final_rotation


init_image = Image.open("/home/kasra/PycharmProjects/reimagine-segmentation/data/image_after_resize/image_after_resize.png/").resize((512, 512), Image.Resampling.LANCZOS)
mask_image = Image.open("/home/kasra/PycharmProjects/reimagine-segmentation/data/image_after_resize/_floor14.jpg/").resize((512, 512), Image.Resampling.LANCZOS)
ref_image_path = "/home/kasra/Downloads/3 (1).jpg/"
reference_image = Image.open(ref_image_path).resize(init_image.size, Image.Resampling.LANCZOS)
control_image = np.array(normal_image(init_image))
a = calculate_masked_normal_vector_pil(normal_map=control_image, mask=mask_image)
b = np.array([0, 0, 1])
r = vector_to_rotation_matrix(a, b)
# Get Euler angles
euler_angles = rotation_matrix_to_euler_angles(r)
print(euler_angles)
rotated_reference = rotate_image(np.array(reference_image), euler_angles[0], euler_angles[1], euler_angles[2])
normal_image(Image.fromarray(rotated_reference)).show()
Image.fromarray(rotated_reference).show()
modified_image = replace_non_masked_part(init_image, mask_image, Image.fromarray(rotated_reference))
modified_image.show()

