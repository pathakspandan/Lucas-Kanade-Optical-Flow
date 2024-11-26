# Temporal Smoothing and Optical Flow Analysis

This repository contains Python scripts for processing image sequences by applying temporal smoothing and computing optical flow using the Lucas-Kanade method.

## Features

1. **Temporal Smoothing:**  
   Smoothens a sequence of image frames using a weighted kernel, enhancing temporal continuity while reducing noise. The smoothed frames are saved as `.npy` files.

2. **Optical Flow Calculation:**  
   Computes the optical flow between consecutive smoothed frames using the Lucas-Kanade method. Outputs include velocity fields (`vx`, `vy`) and reliability scores, saved as `.npy` files.

## Installation

Ensure you have the following Python libraries installed:

- `numpy`
- `scipy`
- `matplotlib`
- `skimage`
- `numba`

You can install the required dependencies via pip if needed.

## Usage

### 1. Temporal Smoothing

This function smooths a sequence of raw image frames. It uses a weighted kernel to smooth frames temporally and stores the smoothed frames as `.npy` files. 

- **Inputs:**  
  - `data_folder`: Path to the folder containing raw image frames.  
  - `new_path`: Path where smoothed `.npy` files will be stored.

### 2. Optical Flow Calculation

This function calculates the optical flow between consecutive smoothed frames using the Lucas-Kanade method. It computes the x and y velocity components (`vx`, `vy`) and reliability scores for each pixel, saving these as `.npy` files. 

- **Inputs:**  
  - `smooth_address`: Path to the folder containing the smoothed `.npy` files.  
  - `save_path`: Path where the optical flow results will be saved.  
  - `sigma`: The spread of Gaussian weights used in the optical flow calculation.  
  - `threshold`: The reliability score threshold for valid velocity fields.

### Outputs

- **Temporal Smoothing:** The smoothed frames are saved as `.npy` files in the `new_path` directory.  
- **Optical Flow:** The computed optical flow velocity components (`vx`, `vy`) and reliability scores are saved as `.npy` files in the `save_path` directory.

## Example Workflow

1. **Step 1:** Temporal smoothing of raw image frames is done by calling the smoothing function with the appropriate paths.  
2. **Step 2:** After smoothing, optical flow is computed by calling the optical flow function on the smoothed frames.

## Contributing

Feel free to fork the repository and create a pull request for any improvements or bug fixes. Please ensure that any contributions are well-documented and come with the appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
