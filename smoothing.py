# Import modules
import os
from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from numba import jit

# Temporal smoothing kernel for frames (Simoncelli kernel)
smooth_kernel = np.array([0.036, 0.249, 0.431, 0.249, 0.036])

@jit(cache=True, parallel=True)
def smoothing(data_folder, file_list, new_path, start_frame=0, end_frame=None, jump=1):
    """
    Applies temporal smoothing to a series of image frames using a predefined kernel.
    
    Parameters:
    - data_folder (str): Path to the folder containing raw image frames.
    - file_list (list): List of filenames in the data folder.
    - new_path (str): Path to save the smoothed frames.
    - start_frame (int, optional): Index of the first frame to process. Default is 0.
    - end_frame (int, optional): Index of the last frame to process. Default is the length of `file_list`.
    - jump (int, optional): Number of frames to skip for smoothing. Default is 1.

    Outputs:
    - Saves temporally smoothed frames as NumPy arrays to the `new_path`.

    Behavior:
    - Handles boundary frames (start and end) separately.
    - For intermediate frames, applies smoothing using up to 5 neighboring frames.
    - Progress is printed every 100 frames.
    """
    if end_frame is None:
        end_frame = len(file_list)

    for k in range(end_frame):
        # Read the current frame
        img1 = io.imread(str(data_folder + '\\' + file_list[k]))

        if k == start_frame:
            # If it's the first frame, use it directly
            test = img1
        elif k == start_frame + jump:
            # Second frame: Average with the previous and next frame
            img2 = io.imread(str(data_folder + '\\' + file_list[k - jump]))
            img3 = io.imread(str(data_folder + '\\' + file_list[k + jump]))
            test = np.average([img2, img1, img3], weights=[0.225, 0.55, 0.225], axis=0)
        elif k == end_frame - 2 * jump:
            # Second last frame: Same as above
            img2 = io.imread(str(data_folder + '\\' + file_list[k - jump]))
            img3 = io.imread(str(data_folder + '\\' + file_list[k + jump]))
            test = np.average([img2, img1, img3], weights=[0.225, 0.55, 0.225], axis=0)
        elif k == end_frame - jump:
            # Last frame: Use it directly
            test = img1
        else:
            # General case: Use 5 frames for smoothing
            img2 = io.imread(str(data_folder + '\\' + file_list[k - jump]))
            img3 = io.imread(str(data_folder + '\\' + file_list[k + jump]))
            img4 = io.imread(str(data_folder + '\\' + file_list[k - 2 * jump]))
            img5 = io.imread(str(data_folder + '\\' + file_list[k + 2 * jump]))
            test = np.average([img4, img2, img1, img3, img5], weights=smooth_kernel, axis=0)

        # Convert to float64 for precision
        test = test.astype(np.float64)
        # Save the smoothed frame
        np.save(os.path.join(new_path, f'{k}.npy'), test)

        # Print progress every 100 frames
        if k % 100 == 0:
            print(k)

# Main implementation with an example
if __name__ == "__main__":
    # Base path for the experiment
    base_path = 'E:\\Spandan\\TTX\\TTX_Experiments_08_22'
    # Address where all temporally smoothed files will be stored
    new_path = os.path.join(base_path, 'smooth')

    # Create the 'smooth' folder if it doesn't exist
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print(f"Folder '{new_path}' created.")
    else:
        print(f"Folder '{new_path}' already exists.")

    # Address where all raw images are
    data_folder = os.path.join(base_path, 'p2')

    # List of all files in the data folder
    file_list = os.listdir(data_folder)

    # Parameters for frame processing
    jump = 1  # Skip amount for frames
    start_frame = 0  # First time-frame
    end_frame = len(file_list)  # Last time-frame

    # Run the smoothing function
    smoothing(data_folder, file_list, new_path, start_frame, end_frame, jump)
