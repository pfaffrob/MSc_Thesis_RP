# Master Thesis Robin Pfaff

This repository contains the contents for my Master's thesis, including all source text files, modules and data analysis scripts.

## Repository Structure

- **Quarto Documentation**: The project uses [Quarto](https://quarto.org) to render markdown documents to HTML and pdf for the thesis write-up
- **Data Analysis**: All relevant data analysis code is sourced from `4_2_scripts/`, which contains Jupyter notebooks for various processing tasks
- **Modules**: The analysis scripts utilize reusable modules from `4_1_modules/`, which include:
  - `config/`: Path configurations and environment setup
  - `lidar/`: LiDAR data processing utilities
  - `raster/`: Raster data manipulation tools
  - `unet_2/`: U-Net deep learning pipeline for image segmentation
  - `utils/`: General helper functions
