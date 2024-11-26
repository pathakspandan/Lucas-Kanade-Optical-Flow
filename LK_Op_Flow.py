## Importing the necessary modules
import skimage.io as io
import numpy as np
import numpy.linalg as nla
import os
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage
from scipy import misc
import scipy.io as sio
import hdf5storage

def LKxOptFlow(img1, img2, sig, thresh):
    """
    Computes optical flow between two image frames using the Lucas-Kanade method.

    Parameters:
    - img1 (ndarray): First time-frame (image 1).
    - img2 (ndarray): Second time-frame (image 2).
    - sig (float): Spread of the Gaussian weights around a pixel.
    - thresh (float): Threshold for reliability score. Velocities below this score are ignored.

    Returns:
    - vx (ndarray): x-component of velocity at each pixel.
    - vy (ndarray): y-component of velocity at each pixel.
    - reliabMat (ndarray): Reliability score matrix for velocities calculated at each pixel.

    Method:
    - Computes gradients of the first frame (spatial and temporal).
    - Uses Gaussian filtering for weighted smoothing of the gradients.
    - Calculates reliability scores and velocity components based on the gradients.
    """
    # Compute spatial and temporal gradients
    ddy = np.gradient(img1, axis=0, edge_order=2)
    ddx = np.gradient(img1, axis=1, edge_order=2)
    dt = img2 - img1

    # Convert to float32 for consistent computation
    ddy = ddy.astype(np.float32)
    ddx = ddx.astype(np.float32)
    dt = dt.astype(np.float32)

    # Apply Gaussian smoothing to gradients
    wdx2 = sp.ndimage.gaussian_filter(ddx**2, sig, mode='nearest')
    wdy2 = sp.ndimage.gaussian_filter(ddy**2, sig, mode='nearest')
    wdxy = sp.ndimage.gaussian_filter(ddx * ddy, sig, mode='nearest')
    wdtx = sp.ndimage.gaussian_filter(ddx * dt, sig, mode='nearest')
    wdty = sp.ndimage.gaussian_filter(ddy * dt, sig, mode='nearest')

    # Calculate determinant and trace for eigenvalue-based reliability
    trace = wdx2 + wdy2
    determinant = (wdx2 * wdy2) - (wdxy**2)
    eps = 1e-6  # Small constant to prevent division by zero

    # Eigenvalues for reliability calculation
    e1 = (trace + np.sqrt(eps + trace**2 - 4 * determinant)) / 2
    e2 = (trace - np.sqrt(eps + trace**2 - 4 * determinant)) / 2
    reliabMat = np.minimum(e1, e2)

    # Calculate velocity components
    vx = ((determinant + eps) ** (-1)) * ((wdxy * wdty) - (wdy2 * wdtx))
    vy = ((determinant + eps) ** (-1)) * ((wdxy * wdtx) - (wdx2 * wdty))

    # Mask velocities below threshold
    vx = vx * (reliabMat > thresh)
    vy = vy * (reliabMat > thresh)
    return vx, vy, reliabMat



# Main implementation
if __name__ == "__main__":
    # Base folder for the experiment
    folder = 'E:\\Spandan\\TTX\\TTX_Experiments_08_22'

    # Path to save optical flow results
    save_path = os.path.join(folder, 'Op_flow')

    # Create the save folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created.")
    else:
        print(f"Folder '{save_path}' already exists.")

    # Path to the folder with temporally smoothed frames
    smooth_address = os.path.join(folder, 'smooth')

    # List of all smoothed files
    file_list = [f for f in sorted(os.listdir(smooth_address)) if f.endswith(".npy")]

    # Parameters for processing
    jump = 1  # Skip amount for time-frames
    start_frame = 0  # Starting frame index
    end_frame = len(file_list)  # Last frame index

    # Optical Flow calculation
    sig = 1.0  # Spread of Gaussian weights
    thresh = 0.01  # Reliability threshold

    for i in range(start_frame, end_frame - jump):
        # Read consecutive frames
        img1_path = os.path.join(smooth_address, file_list[i])
        img2_path = os.path.join(smooth_address, file_list[i + jump])

        # Load frames from .npy files
        try:
            img1 = np.load(img1_path)
            img2 = np.load(img2_path)
        except Exception as e:
            print(f"Error loading frames {i} and {i + jump}: {e}")
        continue

        # Compute optical flow
        vx, vy, reliabMat = LKxOptFlow(img1, img2, sig, thresh)

        # Save results
        flow_save_path = os.path.join(save_path, f"flow_{i}.npz")
        np.savez(flow_save_path, vx=vx, vy=vy, reliability=reliabMat)

        # Print progress
        if i % 10 == 0:
            print(f"Processed optical flow for frame {i} and {i + jump}.")
