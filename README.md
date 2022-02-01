# raster-masker

Data pipeline for downloading opensource raster and vector data for the automatic creation of image and mask tiles for semantic segmentation. Downloads satellite data via Google Earth Engine, building footprints via OpenStreetMap, and automates the labeling of satellite data by burning the vector into the raster to make a mask. Both masks and images are then tiled to specified image size (defaults to (256,256)).

Currently written for downloading USDA NAIP (National Agriculture Imagery Program) raster imagery.

Example workflow for downloading raster and vector data, and creating labelled image masks can be found in [image_labeling.ipynb](https://github.com/NoahChristiansen/raster-masker/blob/main/image_labeling.ipynb).

A demonstration of building a semantic segmentation model (both with and without transfer learning) and training it on tiles generated via this workflow can be found in [image_segmentation.ipynb](https://github.com/NoahChristiansen/raster-masker/blob/main/image_segmentation.ipynb).

The two models trained in the segmentation demonstration notebook were then evaluated on a second set of tiles made from another satellite image (for a different bounding box). The results can be found in [test_segmentation.ipynb](https://github.com/NoahChristiansen/raster-masker/blob/main/test_segmentation.ipynb).
