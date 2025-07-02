"""
This module contains all the functions needed for extracting satellite-derived 
shorelines (SDS)

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
from scipy.spatial import cKDTree

# machine learning modules
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
from shapely.geometry import LineString
import numpy as np
from rasterio.features import rasterize
import rasterio.transform
import pyproj

# other modules
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import gridspec
import pickle
from datetime import datetime, timedelta
from pylab import ginput

# CoastSat modules
from coastsat import SDS_tools, SDS_preprocess

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans
import time
# Main function for batch shoreline detection
def extract_shorelines(metadata, settings):
    """
    Main function to extract shorelines from satellite images
    """
    start_total = time.time()

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    filepath_models = os.path.join(os.getcwd(), 'classification', 'models')
    output = dict([])
    filepath_jpg = os.path.join(filepath_data, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    plt.close('all')

    print('Mapping shorelines:')
    default_min_length_sl = settings['min_length_sl']

    # Cache to avoid recomputing identical shoreline buffers
    buffer_cache = {}

    for satname in metadata.keys():
        sat_start = time.time()

        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        output_timestamp = []
        output_shoreline = []
        output_filename = []
        output_cloudcover = []
        output_geoaccuracy = []
        output_idxkeep = []
        output_t_mndwi = []

        str_new = ''
        if not sklearn.__version__[:4] == '0.20':
            str_new = '_new'

        load_model_start = time.time()
        if satname in ['L5','L7','L8','L9']:
            pixel_size = 15
            if settings['sand_color'] == 'dark':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_dark%s.pkl'%str_new))
            elif settings['sand_color'] == 'bright':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_bright%s.pkl'%str_new))
            elif settings['sand_color'] == 'default':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat%s.pkl'%str_new))
            elif settings['sand_color'] == 'latest':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_latest%s.pkl'%str_new))   
        elif satname == 'S2':
            pixel_size = 10
            clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_S2%s.pkl'%str_new))
        print(f"Model loading took {time.time() - load_model_start:.2f} seconds")

        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        settings['min_length_sl'] = 200 if satname == 'L7' else default_min_length_sl

        for i in range(len(filenames)):
            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            prep_start = time.time()
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(
                fn, satname, settings['cloud_mask_issue'], settings['pan_off'], settings['s2cloudless_prob'])
            print(f"  | Preprocessing: {time.time() - prep_start:.2f}s", end='')

            image_epsg = metadata[satname]['epsg'][i]

            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                             (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99:
                continue
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            if cloud_cover > settings['cloud_thresh']:
                continue

            buffer_start = time.time()
            buffer_key = (tuple(cloud_mask.shape), tuple(np.round(georef, 6)))

            if buffer_key in buffer_cache:
                print("  | Buffer (cached)", end='')
                im_ref_buffer = buffer_cache[buffer_key]
            else:
                print("  | Buffer (computed)", end='')
                im_ref_buffer = create_shoreline_buffer(cloud_mask.shape, georef, image_epsg, settings)
                buffer_cache[buffer_key] = im_ref_buffer
            print(f"  | Buffer: {time.time() - buffer_start:.2f}s", end='')

            classify_start = time.time()
            im_classif, im_labels = classify_image_NN(im_ms, cloud_mask, min_beach_area_pixels, clf)
            print(f"  | Classify: {time.time() - classify_start:.2f}s", end='')

            map_start = time.time()
            if settings['adjust_detection']:
                date = filenames[i][:19]
                skip_image, shoreline, t_mndwi = adjust_detection(im_ms, cloud_mask, im_nodata, im_labels,
                                                                  im_ref_buffer, image_epsg, georef,
                                                                  settings, date, satname)
                if skip_image:
                    continue
            else:
                try:
                    if sum(im_labels[im_ref_buffer,0]) < 50:
                        im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                        contours_mwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
                    else:
                        contours_mwi, t_mndwi = find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer)
                except Exception as e:
                    print(f'Could not map shoreline for this image: {filenames[i]}, reason: {e}')
                    continue
            print(f"  | Map shoreline: {time.time() - map_start:.2f}s")

            process_start = time.time()
            shoreline = process_shoreline(contours_mwi, cloud_mask_adv, im_nodata,
                                          georef, image_epsg, settings)
            print(f"  | Process: {time.time() - process_start:.2f}s")

            if settings['check_detection'] or settings['save_figure']:
                date = filenames[i][:19]
                if not settings['check_detection']:
                    plt.ioff()
                skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                            image_epsg, georef, settings, date, satname)
                if skip_image:
                    continue

            output_timestamp.append(metadata[satname]['dates'][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            output_t_mndwi.append(t_mndwi)

        output[satname] = {
            'dates': output_timestamp,
            'shorelines': output_shoreline,
            'filename': output_filename,
            'cloud_cover': output_cloudcover,
            'geoaccuracy': output_geoaccuracy,
            'idx': output_idxkeep,
            'MNDWI_threshold': output_t_mndwi,
        }
        print(f"\n{satname} batch took {time.time() - sat_start:.2f} seconds\n")

    if plt.get_fignums():
        plt.close()

    output = SDS_tools.merge_output(output)

    filepath = filepath_data
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    print(f"Total extraction time: {time.time() - start_total:.2f} seconds")
    return output
###################################################################################################
# IMAGE CLASSIFICATION FUNCTIONS
###################################################################################################

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    The features include spectral normalized-difference indices and standard 
    deviation of the image for all the bands and indices.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # NIR-G
    im_NIRG = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G
    im_SWIRG = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = SDS_tools.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  SDS_tools.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    im_std = SDS_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features

def classify_image_NN(im_ms, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier

    Returns:    
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_inf = np.any(np.isinf(vec_features), axis=1)    
    vec_mask = np.logical_or(vec_cloud,np.logical_or(vec_nan,vec_inf))
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1)

    return im_classif, im_labels

###################################################################################################
# CONTOUR MAPPING FUNCTIONS
###################################################################################################

def find_wl_contours1(im_ndwi, cloud_mask, im_ref_buffer):
    """
    Traditional method for shoreline detection using a global threshold.
    Finds the water line by thresholding the Normalized Difference Water Index 
    and applying the Marching Squares Algorithm to contour the iso-value 
    corresponding to the threshold.

    KV WRL 2018

    Arguments:
    -----------
    im_ndwi: np.ndarray
        Image (2D) with the NDWI (water index)
    cloud_mask: np.ndarray
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline

    Returns:    
    -----------
    contours: list of np.arrays
        contains the coordinates of the contour lines
    t_mwi: float
        Otsu threshold used to map the contours

    """
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]
    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer,se)
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    # reshape spectral index image to vector
    vec_ndwi = im_ndwi.reshape(nrows*ncols)
    # keep pixels that are in the buffer and not in the cloud mask
    vec_mask = cloud_mask.reshape(nrows*ncols)
    vec = vec_ndwi[np.logical_and(vec_buffer,~vec_mask)]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    im_ndwi_buffer = np.copy(im_ndwi)
    im_ndwi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndwi_buffer, t_otsu)
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = process_contours(contours)

    return contours, t_otsu

def find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer):
    """
    New robust method for extracting shorelines. Incorporates the classification
    component to refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline

    Returns:    
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        MNDWI (Modified Normalized Difference Water Index) image
    t_mwi: float
        Otsu sand/water threshold used to map the contours

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)

    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer, se)
    # create a buffer around the sandy beach
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    
    # select water/sand pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
    int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_sand) > 0:
        if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
            int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
        else:
            int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_water,int_sand, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])
    t_wi = filters.threshold_otsu(int_all[:,1])

    # find contour with Marching-Squares algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_ref_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)
    # remove contour points that are NaNs (around clouds)
    contours_wi = process_contours(contours_wi)
    contours_mwi = process_contours(contours_mwi)

    # only return MNDWI contours and threshold
    return contours_mwi, t_mwi

###################################################################################################
# SHORELINE PROCESSING FUNCTIONS
###################################################################################################

# def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings):
#     """
#     Creates a buffer around each reference shoreline. The size of the buffer is 
#     given by settings['max_dist_ref'].

#     Arguments:
#     -----------
#     im_shape: np.array
#         size of the image (rows, columns)
#     georef: np.array
#         vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
#     image_epsg: int
#         spatial reference system of the image from which the contours were extracted
#     pixel_size: int
#         size of the pixel in meters (15 for Landsat, 10 for Sentinel-2)
#     settings: dict with the following keys
#         'output_epsg': int
#             output spatial reference system
#         'reference_shoreline': list of np.array
#             list of coordinate arrays for each reference shoreline
#         'max_dist_ref': int
#             maximum distance from the reference shoreline in meters

#     Returns:    
#     -----------    
#     im_buffer: np.array
#         binary image, True where the buffer is, False otherwise

#     """
#     # Initialize the image buffer
#     im_buffer = np.zeros(im_shape, dtype=bool)

#     if 'reference_shoreline' in settings.keys():
#         max_dist_ref_pixels = np.ceil(settings['max_dist_ref'] / pixel_size)
#         se = morphology.disk(max_dist_ref_pixels)

#         for ref_sl in settings['reference_shoreline']:
#             # Convert each shoreline to pixel coordinates
#             ref_sl_conv = SDS_tools.convert_epsg(ref_sl, settings['output_epsg'], image_epsg)
#             ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
#             ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)

#             # Ensure coordinates are within image bounds
#             idx_row = np.logical_and(ref_sl_pix_rounded[:, 0] > 0, ref_sl_pix_rounded[:, 0] < im_shape[1])
#             idx_col = np.logical_and(ref_sl_pix_rounded[:, 1] > 0, ref_sl_pix_rounded[:, 1] < im_shape[0])
#             idx_inside = np.logical_and(idx_row, idx_col)
#             ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside, :]

#             # Create a binary image for each shoreline
#             im_binary = np.zeros(im_shape, dtype=bool)
#             for j in range(len(ref_sl_pix_rounded)):
#                 im_binary[ref_sl_pix_rounded[j, 1], ref_sl_pix_rounded[j, 0]] = True

#             # Dilate the binary image to create a buffer around the reference shoreline
#             im_buffer = np.logical_or(im_buffer, morphology.binary_dilation(im_binary, se))

#     return im_buffer

def create_shoreline_buffer(im_shape, georef, image_epsg, settings):
    """
    Creates a binary mask representing a buffer zone around each reference shoreline. 
    This buffer is computed using Shapely and rasterized using Rasterio for performance.
    
    Arguments:
    -----------
    im_shape: np.array
        Shape of the image (rows, columns)
    georef: np.array
        Georeferencing transform [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        EPSG code of the image's coordinate reference system
    settings: dict with the following keys
        'output_epsg': int
            EPSG code of the reference shoreline coordinate system
        'reference_shoreline': list of np.array
            List of arrays with coordinates for each reference shoreline
        'max_dist_ref': int
            Buffer radius in meters to apply around the reference shorelines

    Returns:
    -----------
    im_buffer: np.array (dtype=bool)
        Binary mask where True indicates buffered shoreline area and False elsewhere
    """

    # Default to empty binary mask
    im_buffer = np.zeros(im_shape, dtype='uint8')

    if 'reference_shoreline' not in settings:
        return im_buffer.astype(bool)

    # Define transform from pixel to world
    x_origin, x_res, _, y_origin, _, y_res = georef
    transform_affine = rasterio.transform.Affine(x_res, 0, x_origin, 0, y_res, y_origin)

    max_dist = settings['max_dist_ref']  # buffer size in meters
    proj = pyproj.Transformer.from_crs(settings['output_epsg'], image_epsg, always_xy=True)

    shapes = []
    for ref_sl in settings['reference_shoreline']:
        if len(ref_sl) < 2:
            continue  # skip degenerate lines
        try:
            ref_sl_conv = np.array(proj.transform(ref_sl[:, 0], ref_sl[:, 1])).T
            line = LineString(ref_sl_conv)
            buffered = line.buffer(max_dist)  # buffer in image CRS units
            shapes.append(buffered)
        except Exception as e:
            print(f"[Warning] Could not buffer reference shoreline: {e}")

    if shapes:
        # Rasterize buffered geometries into binary mask with dtype uint8
        im_buffer = rasterize(
            [(shape, 1) for shape in shapes],
            out_shape=im_shape,
            transform=transform_affine,
            fill=0,
            dtype='uint8'
        )

    # Convert to bool to maintain compatibility with rest of workflow
    return im_buffer.astype(bool)


def process_contours(contours):
    """
    Remove contours that contain NaNs, usually these are contours that are in contact 
    with clouds.
    
    KV WRL 2020
    
    Arguments:
    -----------
    contours: list of np.array
        image contours as detected by the function skimage.measure.find_contours    
    
    Returns:
    -----------
    contours: list of np.array
        processed image contours (only the ones that do not contains NaNs) 
        
    """
    
    # initialise variable
    contours_nonans = []
    # loop through contours and only keep the ones without NaNs
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    
    return contours_nonans
    
# def process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings):
#     """
#     Converts the contours from image coordinates to world coordinates. This function also removes the contours that:
#         1. are too small to be a shoreline (based on the parameter settings['min_length_sl'])
#         2. are too close to cloud pixels (based on the parameter settings['dist_clouds'])
#         3. are adjacent to noData pixels
    
#     KV WRL 2018

#     Arguments:
#     -----------
#         contours: np.array or list of np.array
#             image contours as detected by the function find_contours
#         cloud_mask: np.array
#             2D cloud mask with True where cloud pixels are
#         im_nodata: np.array
#             2D mask with True where noData pixels are
#         georef: np.array
#             vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
#         image_epsg: int
#             spatial reference system of the image from which the contours were extracted
#         settings: dict
#             contains the following fields:
#         output_epsg: int
#             output spatial reference system
#         min_length_sl: float
#             minimum length of shoreline perimeter to be kept (in meters)
#         dist_clouds: int
#             distance in metres defining a buffer around cloudy pixels where the shoreline cannot be mapped

#     Returns:    
#     -----------
#         shoreline: np.array
#             array of points with the X and Y coordinates of the shoreline

#     """

#     # convert pixel coordinates to world coordinates
#     contours_world = SDS_tools.convert_pix2world(contours, georef)
#     # convert world coordinates to desired spatial reference system
#     contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    
#     # 1. Remove contours that have a perimeter < min_length_sl (provided in settings dict)
#     # this enables to remove the very small contours that do not correspond to the shoreline
#     contours_long = []
#     for l, wl in enumerate(contours_epsg):
#         coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
#         a = LineString(coords) # shapely LineString structure
#         if a.length >= settings['min_length_sl']:
#             contours_long.append(wl)
#     # format points into np.array
#     x_points = np.array([])
#     y_points = np.array([])
#     for k in range(len(contours_long)):
#         x_points = np.append(x_points,contours_long[k][:,0])
#         y_points = np.append(y_points,contours_long[k][:,1])
#     contours_array = np.transpose(np.array([x_points,y_points]))
    
#     shoreline = contours_array
    
#     # 2. Remove any shoreline points that are close to cloud pixels (effect of shadows)
#     if sum(sum(cloud_mask)) > 0:
#         # get the coordinates of the cloud pixels
#         idx_cloud = np.where(cloud_mask)
#         idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
#         # convert to world coordinates and same epsg as the shoreline points
#         coords_cloud = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(idx_cloud, georef),
#                                                image_epsg, settings['output_epsg'])
#         # only keep the shoreline points that are at least 30m from any cloud pixel
#         idx_keep = np.ones(len(shoreline)).astype(bool)
#         for k in range(len(shoreline)):
#             if np.any(np.linalg.norm(shoreline[k,:] - coords_cloud, axis=1) < settings['dist_clouds']):
#                 idx_keep[k] = False     
#         shoreline = shoreline[idx_keep] 
        
#     # 3. Remove any shoreline points that are attached to nodata pixels
#     if sum(sum(im_nodata)) > 0:
#         # get the coordinates of the cloud pixels
#         idx_cloud = np.where(im_nodata)
#         idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
#         # convert to world coordinates and same epsg as the shoreline points
#         coords_cloud = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(idx_cloud, georef),
#                                                image_epsg, settings['output_epsg'])
#         # only keep the shoreline points that are at least 30m from any nodata pixel
#         idx_keep = np.ones(len(shoreline)).astype(bool)
#         for k in range(len(shoreline)):
#             if np.any(np.linalg.norm(shoreline[k,:] - coords_cloud, axis=1) < 30):
#                 idx_keep[k] = False     
#         shoreline = shoreline[idx_keep] 

#     return shoreline



def process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings):
    """
    Processes detected shoreline contours by converting them to world coordinates and 
    applying filters based on shoreline length, cloud proximity, and nodata proximity.

    Arguments:
    -----------
    contours: list of np.array
        List of shoreline contours in pixel coordinates
    cloud_mask: np.array (dtype=bool)
        Binary mask where True indicates cloud-covered pixels
    im_nodata: np.array (dtype=bool)
        Binary mask where True indicates nodata pixels (e.g., masked water or artifacts)
    georef: np.array
        Georeferencing transform [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        EPSG code of the image’s native coordinate reference system
    settings: dict with the following keys
        'output_epsg': int
            Desired output coordinate system for shoreline points
        'min_length_sl': float
            Minimum contour length in meters to be considered valid
        'dist_clouds': float
            Minimum allowed distance from cloud pixels in meters

    Returns:
    -----------
    shoreline: np.array
        Array of shoreline points (x, y) in output CRS, after filtering
    """

    # Step 1: Convert contours to world coordinates and reproject
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])

    # Step 2: Remove short contours
    contours_long = [wl for wl in contours_epsg
                     if LineString([(x, y) for x, y in wl]).length >= settings['min_length_sl']]

    # Combine long contours into single array
    if not contours_long:
        return np.empty((0, 2))
    shoreline = np.vstack(contours_long)

    # Step 3: Remove shoreline points close to cloud pixels
    if cloud_mask.any():
        cloud_idx = np.column_stack(np.where(cloud_mask))
        cloud_coords = SDS_tools.convert_pix2world(cloud_idx, georef)
        cloud_coords = SDS_tools.convert_epsg(cloud_coords, image_epsg, settings['output_epsg'])
        cloud_tree = cKDTree(cloud_coords)
        keep = cloud_tree.query(shoreline, distance_upper_bound=settings['dist_clouds'])[0] == np.inf
        shoreline = shoreline[keep]

    # Step 4: Remove shoreline points close to nodata pixels
    if im_nodata.any():
        nodata_idx = np.column_stack(np.where(im_nodata))
        nodata_coords = SDS_tools.convert_pix2world(nodata_idx, georef)
        nodata_coords = SDS_tools.convert_epsg(nodata_coords, image_epsg, settings['output_epsg'])
        nodata_tree = cKDTree(nodata_coords)
        keep = nodata_tree.query(shoreline, distance_upper_bound=30)[0] == np.inf
        shoreline = shoreline[keep]

    return shoreline


###################################################################################################
# INTERACTIVE/PLOTTING FUNCTIONS
###################################################################################################

def show_detection(im_ms, cloud_mask, im_labels, shoreline,image_epsg, georef,
                   settings, date, satname):
    """
    Shows the detected shoreline to the user for visual quality control. 
    The user can accept/reject the detected shorelines  by using keep/skip
    buttons.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, 'jpg_files', 'detection')

    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)

    cmap = plt.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]

    # compute MNDWI grayscale image
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)

    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg), georef)
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])

    if plt.get_fignums():
        # get open figure if it exists
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]     

    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        if settings['check_detection']:
            try:
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
            except Exception:
                pass

        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # horizontal subplots
            gs = gridspec.GridSpec(4, 1,height_ratios=[0.1,1,1,1])
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[3], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[0])
        else:
            # vertical subplots
            gs = gridspec.GridSpec(2, 3,height_ratios=[1,8],hspace=0.1)
            gs.update(bottom=0.03, top=0.99, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[1,0])
            ax2 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[1,2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[0,:])
            
    # add timeline on top
    date_start = datetime.strptime(settings['inputs']['dates'][0],'%Y-%m-%d')
    date_end = datetime.strptime(settings['inputs']['dates'][1],'%Y-%m-%d')
    ax4.axis('off')
    ax4.axhline(y=0,ls='-',lw=2,c='k')
    ax4.set(xlim=[date_start-timedelta(days=30),date_end+timedelta(days=30)],ylim=[-0.1,0.1])
    for k in range(date_start.year,date_end.year+1):
        ax4.plot(datetime(k,1,1),0,'ko',ms=6)
        ax4.text(datetime(k,1,1),-0.05,str(k)[-2:],ha='center',va='center')
    ax4.plot(datetime.strptime(date[:10],'%Y-%m-%d'),0,'rs',ms=10,mec='k')
    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=12)

    # create image 2 (classification)
    ax2.imshow(im_class)
    ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title(date, fontweight='bold', fontsize=12)

    # create image 3 (MNDWI)
    ax3.imshow(im_mwi, cmap='bwr')
    ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    ax3.set_title(satname, fontweight='bold', fontsize=12)
    
    # additional options
    ax1.set_anchor('W')
    ax2.set_anchor('C')
    ax3.set_anchor('E')
    
    # add colorbar for MNDWI
    # cb = plt.colorbar(mwi_plot)
    # cb.ax.tick_params(labelsize=10)
    # cb.set_label('MNDWI values')

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings['check_detection']:

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)
        plt.close(fig)

    else:
        # don't close the figure window, but remove all axes and settings, ready for next plot
        for ax in fig.axes:
            ax.clear()

    return skip_image

def adjust_detection(im_ms, cloud_mask, im_nodata, im_labels, im_ref_buffer, image_epsg, georef,
                     settings, date, satname):
    """
    Advanced version of show detection where the user can adjust the detected 
    shorelines with a slide bar.

    KV WRL 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline 
    t_mndwi: float
        value of the MNDWI threshold used to map the shoreline

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, 'jpg_files', 'detection')
    # format date
    date_str = datetime.strptime(date,'%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d  %H:%M:%S')
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)
    cmap = plt.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]

    # compute MNDWI grayscale image
    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # buffer MNDWI using reference shoreline
    im_mndwi_buffer = np.copy(im_mndwi)
    im_mndwi_buffer[~im_ref_buffer] = np.nan

    # get MNDWI pixel intensity in each class (for histogram plot)
    int_sand = im_mndwi[im_labels[:,:,0]]
    int_ww = im_mndwi[im_labels[:,:,1]]
    int_water = im_mndwi[im_labels[:,:,2]]
    labels_other = np.logical_and(np.logical_and(~im_labels[:,:,0],~im_labels[:,:,1]),~im_labels[:,:,2])
    int_other = im_mndwi[labels_other]
    
    # create figure
    if plt.get_fignums():
            # if it exists, open the figure 
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            ax4 = fig.axes[3]      
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4,1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1,:])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # plot image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis('off')
    ax1.set_title('%s - %s'%(sitename, satname), fontsize=12)

    # plot image 2 (classification)
    ax2.imshow(im_class)
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1.1, 0.5), fontsize=10)
    ax2.set_title(date_str, fontsize=12)

    # plot image 3 (MNDWI)
    ax3.imshow(im_mndwi, cmap='bwr')
    ax3.axis('off')
    ax3.set_title('MNDWI', fontsize=12)
    
    # plot histogram of MNDWI values
    binwidth = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='PDF',yticklabels=[], xlim=[-1,1])
    if len(int_sand) > 0 and sum(~np.isnan(int_sand)) > 0:
        bins = np.arange(np.nanmin(int_sand), np.nanmax(int_sand) + binwidth, binwidth)
        ax4.hist(int_sand, bins=bins, density=True, color=colours[0,:], label='sand')
    if len(int_ww) > 0 and sum(~np.isnan(int_ww)) > 0:
        bins = np.arange(np.nanmin(int_ww), np.nanmax(int_ww) + binwidth, binwidth)
        ax4.hist(int_ww, bins=bins, density=True, color=colours[1,:], label='whitewater', alpha=0.75) 
    if len(int_water) > 0 and sum(~np.isnan(int_water)) > 0:
        bins = np.arange(np.nanmin(int_water), np.nanmax(int_water) + binwidth, binwidth)
        ax4.hist(int_water, bins=bins, density=True, color=colours[2,:], label='water', alpha=0.75) 
    if len(int_other) > 0 and sum(~np.isnan(int_other)) > 0:
        bins = np.arange(np.nanmin(int_other), np.nanmax(int_other) + binwidth, binwidth)
        ax4.hist(int_other, bins=bins, density=True, color='C4', label='other', alpha=0.5) 
    
    # automatically map the shoreline based on the classifier if enough sand pixels
    try:
        if sum(sum(im_labels[:,:,0])) > 50:
            # use classification to refine threshold and extract the sand/water interface
            contours_mndwi, t_mndwi = find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer)
        else:       
            # find water contours on MNDWI grayscale image
            contours_mndwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)    
    except:
        print('Could not map shoreline so image was skipped')
        # clear axes and return skip_image=True, so that image is skipped above
        for ax in fig.axes:
            ax.clear()
        return True,[],[]

    # process the water contours into a shoreline
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata) 
    shoreline = process_shoreline(contours_mndwi, cloud_mask_adv, im_nodata, georef, image_epsg, settings)
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg), georef)
    else: sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    t_line = ax4.axvline(x=t_mndwi,ls='--', c='k', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw() # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title('Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
    while True:  
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0: 
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(pt[0][0]) >= 1: continue
            # update the threshold value
            t_mndwi = pt[0][0]
            # update the plot
            t_line.set_xdata([t_mndwi,t_mndwi])
            # map contours with new threshold
            contours = measure.find_contours(im_mndwi_buffer, t_mndwi)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours) 
            # process the water contours into a shoreline
            shoreline = process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings)
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg), georef)
            else: sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:,0], sl_pix[:,1]])
            sl_plot2[0].set_data([sl_pix[:,0], sl_pix[:,1]])
            sl_plot3[0].set_data([sl_pix[:,0], sl_pix[:,1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break
    
    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key
    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_skip.remove()
        btn_keep.remove()
        btn_esc.remove()

        # keep/skip image according to the pressed key, 'escape' to break the loop
        if key_event.get('pressed') == 'right':
            skip_image = False
            break
        elif key_event.get('pressed') == 'left':
            skip_image = True
            break
        elif key_event.get('pressed') == 'escape':
            plt.close()
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline, t_mndwi
