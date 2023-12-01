# Supervised Monocular Depth Estimation

Monocular depth estimation using a residual UNet with the [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

View the rendered notebook on nbviewer [here](https://nbviewer.org/github/supperted825/depth-estimation/blob/main/train.ipynb).

The objective function is a combination of the smoothness loss, structural similarity (SSIM) loss, and a regression loss (RMSE) between the produced depth map and the truth depth map. L1 loss was initially used for regression loss, but the quality of predictions increased significantly after switching to RMSE.

This project was complted as part of Red Dragon AI's Advanced Computer Vision course.


