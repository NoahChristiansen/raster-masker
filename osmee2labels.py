import numpy as np
import osmnx as ox
from PIL import Image
from osgeo import gdal, ogr
import rasterio as rio
from rasterio.features import bounds
from rasterio import windows
from geojson import dump
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geemap
import os
from itertools import product
from shapely.errors import ShapelyDeprecationWarning
from tqdm import tqdm
import ee

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

ox.config(timeout=1000)
ee.Initialize()

def get_NAIP_raster_from_bbox(bbox):
    """
    Query EarthEngine for the USDA/NAIP/DOQQ ImageCollection, specifying the area based on the bounding box provided. Sort results by CLOUDY_PIXEL_PERCENTAGE and return the first Image in the collection.
    
    Parameters
    ----------
    
    bbox: tuple, of the form (west, south, east, north)
        A tuple containing the bounding box to query the ImageCollection for. Format should be (west, south, east, north)
    
    """
    
    return (ee.ImageCollection('USDA/NAIP/DOQQ')
            .filterBounds(ee.Geometry.BBox(*bbox))
            .sort('CLOUDY_PIXEL_PERCENTAGE')
            .first())

def get_Image_bbox(rasterImage:ee.Image):
    """
    Returns bounding box of input Image
    
    Parameters
    ----------
    
    rasterImage: ee.Image
            Google Earth Engine Image file to get a bounding box for.
    """
    bands = [band['id'] for band in rasterImage.getInfo()['bands']]
    
    bbox_coords = rasterImage.select(bands[0]).getInfo()["properties"]['system:footprint']['coordinates']
    
    pixel_dims = rasterImage.arrayDimensions().getInfo()['bands'][0]['dimensions']
    
    #print(bbox_coords)
    print(f"Pixel Dimensions: {pixel_dims}")
    print(f"Bands: {bands}")
    return rio.features.bounds(Polygon(bbox_coords))

def subdivide_bbox(total_bbox, tiles_x = 4, tiles_y = 4):
    """
    Takes a bounding box tuple (west, south, east, north) as input, along with the requested number of tiles in both the x and y directions.
    Returns a list of tiles_x*tiles_y bounding boxes (as tuples) which subdivide the original bounding box.
    
    Parameters
    ----------
    
    total_bbox: tuple, of the form (west, south, east, north)
            A tuple containing the bounding box to be subdivided. Tuple should contain four values formatted as (west, south, east, north).
            
    tiles_x: int
            The number of tiles to make in the x-direction (east-west).
    
    tiles_y: int
            The number of tiles to make in the y_direction (north-south).
    
    """
    w,s,e,n = total_bbox
    sub_bboxs = []
    for i in range(tiles_y):
        for j in range(tiles_x):
            new_n = n-(i*(n-s)*(1/tiles_y))
            new_s = n-((i+1)*(n-s)*(1/tiles_y))
            new_e = w + ((j+1)*(e-w)*(1/tiles_x))
            new_w = w + (j*(e-w)*(1/tiles_x))
            sub_bboxs.append((new_w, new_s, new_e, new_n))
    return sub_bboxs

def export_subdivided_raster_vector(raster_image:ee.Image, bbox_list, raster_path_out:str, vector_path_out:str, bands = None):
    """
    Takes a raster image (as an ee.Image), and a list of bounding boxs that subdivide the raster.
    For each of the bounding boxes, the raster is clipped to that bounding box and a corresponding GeoDataFrame of building footprints for the same area is pulled from OpenStreetMap (OSM).
    The clipped raster and corresponding GeoDataFrame of building footprints are then output to the raster_path_out and vector_path_out folders (raster is saved as .tif, GeoDataFrame is saved as .geojson).
    
    Parameters
    ----------
    
    raster_image: ee.Image
            An Image file from Google Earth Engine to be subdivided for output as .tif files.
    
    bbox_list: list
            A list of bounding boxes that subdivide the bounding box of raster_image.
            
    raster_path_out: str
            Directory to save the clipped raster files to (as .tif files).
    
    vector_path_out: str
            Directory to save the building footprints vector file to (as .geojson files).
    """
    
    os.makedirs(raster_path_out,exist_ok=True)
    os.makedirs(vector_path_out,exist_ok=True)
    
    for i in tqdm(range(len(bbox_list))):
        ## Clip Raster to tile bounding box
        subset = raster_image.clip(ee.Geometry.BBox(*bbox_list[i]))
        
        if bands == None:
            bands = [band['id'] for band in subset.getInfo()['bands']]
        
        if len(bands) > 3:
            bands.remove("N")
            bands = bands[:3]
        ## Get the bounding box for the tile (in case this differs from the original tile bounding box)
        subset_bounds = rio.features.bounds(Polygon(subset.select(bands[0]).getInfo()["properties"]['system:footprint']['coordinates'][0]))
        ## Query OpenStreetMap for building footprints within the tile bounding box
        subset_gdf = geemap.osm_gdf_from_bbox(north = subset_bounds[3], south = subset_bounds[1],
                                              east = subset_bounds[2], west = subset_bounds[0],
                                              tags = {"building":True})
        
        #print(subset.)

        ## Save the raster tile to the tiles folder
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            warnings.simplefilter('ignore', category=ShapelyDeprecationWarning)
            geemap.ee_export_image(subset.select(*bands), f"{raster_path_out}/tile_{i}.tif", scale = 0.6);
        ## Save the OSM building footprints as GeoJSON to the tiles folder
        with open(f"{vector_path_out}/tile_{i}.geojson", "w") as f:
            dump(subset_gdf, f, allow_nan=True)
            
def create_poly_mask(raster_src, vector_src, dest_name = "", burn_value = 255):
    """
    Function to create single image mask, given a raster image and a geojson with masks.
    
    Parameters
    ----------
    
    raster_src: str
            Filepath for raster image file (as .tif).
    
    vector_src: str
            Filepath for vector file of polygons to mask (as .geojson).
            
    dest_name: str
            Path to save mask .tif file to.
    
    burn_value: int, default is 255
            Value to use when burning polygons into raster file.
    """
    
    
    ## Open geojson file with building polygons
    src_vect = ogr.Open(vector_src)
    src_layer = src_vect.GetLayer()

    ## Open raster image (tif file) that corresponds to geojson file
    src_rast = gdal.Open(raster_src)
    cols = src_rast.RasterXSize
    rows = src_rast.RasterYSize

    mem_layer = gdal.GetDriverByName("GTiff")
    dest_ds = mem_layer.Create(dest_name, cols, rows, 1, gdal.GDT_Byte,
                               options = ["COMPRESS=LZW"])

    dest_ds.SetGeoTransform(src_rast.GetGeoTransform())
    dest_ds.SetProjection(src_rast.GetProjection())
    band = dest_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    gdal.RasterizeLayer(dest_ds, [1], src_layer, burn_values = [255])
    dest_ds = 0

    mask = np.array(Image.open(dest_name))
    
    return mask

def build_masks(source_raster_directory, source_vector_directory, mask_destination):
    """
    Function to create masks given a directory of raster images (.tif files) and a directory of vector files containing masks (.geojson files).
    Calls `create_poly_mask` on each image-mask pair.
    
    Parameters
    ----------
    
    source_raster_directory: str
            Path to directory of raster (.tif) files.
            
    source_vector_directory: str
            Path to directory of vector (.geojson) files
    
    mask_destination: str
            Path to directory to save mask (.tif) files to.
    """
    
    os.makedirs(mask_destination, exist_ok=True)
    
    raster_paths = sorted([source_raster_directory+file for file in os.listdir(source_raster_directory)
                           if file.endswith(".tif")])

    vector_paths = sorted([source_vector_directory+file for file in os.listdir(source_vector_directory)
                           if file.endswith(".geojson")])
    
    for img,lbl in tqdm(zip(raster_paths, vector_paths)):
        output_path = mask_destination+lbl.split("/")[-1].replace("geojson","tif")
        create_poly_mask(raster_src=img, vector_src=lbl, dest_name=output_path)
        

def get_tiles(ds, width=256, height=256):
    """
    Uses rasterio's Window function to divide a raster image into tiles with specified width and height. Returns a generator with windows and transforms.
    
    Parameters
    ----------
    
    ds: rasterio.io.DatasetReader
        Raster image (.tif) file read in using rasterio.open()
        
    width: int
        Width of tile (in pixels)
        
    height: int
        Height of tile (in pixels)
        
    *Inspiration for this approach comes from: https://gis.stackexchange.com/a/290059
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
        
def tile_tifs(raster_paths:list, mask_paths:list, raster_tiles_out:str, mask_tiles_out:str, tile_size = (256,256)):
    """
    Takes input lists of file paths to raster images and corresponding mask images (saved as .tif files), and outputs tiles of the specified size for both rasters and masks to the corresponding output directories.
    
    Parameters
    ----------
    
    raster_paths: list, or other iterable
            A list of file paths for raster images (.tif files) to make tiles from.
    
    mask_paths: list, or other iterable
            A list of file paths for mask images (.tif files) to make tiles from.
            
    raster_tiles_out: str
            Name of directory to save raster tile outputs to.
            
    mask_tiles_out: str
            Name of directory to save mask tile outputs to.
    
    tile_size: tuple, defaults to (256,256)
            Tuple of two values corresponding to desired height and width of tiles.
            Default value is (256,256)
            
            
    *Inspiration for this approach comes from: https://gis.stackexchange.com/a/290059
    """
    
    os.makedirs(raster_tiles_out, exist_ok=True)
    os.makedirs(mask_tiles_out, exist_ok=True)

    for raster_pth, mask_pth in tqdm(zip(raster_paths,mask_paths)):
        raster_tile = rio.open(raster_pth)
        mask_tile = rio.open(mask_pth)

        t_width, t_height = tile_size
        r_meta = raster_tile.meta.copy()
        m_meta = mask_tile.meta.copy()
        
        tile_no = raster_pth.split("_")[1].replace(".tif","")

        for (r_window,r_transform),(m_window, m_transform) in zip(get_tiles(raster_tile), get_tiles(mask_tile)):
            r_meta['transform'] = r_transform
            r_meta['width'], r_meta['height'] = r_window.width, r_window.height

            m_meta['transform'] = m_transform
            m_meta['width'], m_meta['height'] = m_window.width, m_window.height

            r_outpath = os.path.join(raster_tiles_out,f"tile_{tile_no}_{int(r_window.col_off)}-{int(r_window.row_off)}.tif")
            with rio.open(r_outpath, 'w', **r_meta) as r_outds:
                r_outds.write(raster_tile.read(window=r_window))

            m_outpath = os.path.join(mask_tiles_out,f"tile_{tile_no}_{int(m_window.col_off)}-{int(m_window.row_off)}.tif")
            with rio.open(m_outpath, 'w', **m_meta) as m_outds:
                m_outds.write(mask_tile.read(window=m_window))