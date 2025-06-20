# ==========================================================#
# Shoreline extraction from satellite images                #
# ==========================================================#
import sys
print("Running under:", sys.executable)
import os
import numpy as np
import pickle
import warnings
import gc
import argparse
import json
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    matplotlib.use('Qt5Agg')
except ImportError:
    print('Qt5Agg not available, using default backend.')
# plt.ion()
import geopandas as gpd
from shapely.geometry import MultiLineString
from shapely.geometry import Polygon
import pandas as pd
from scipy import interpolate
from scipy import stats
from datetime import datetime, timedelta
import pytz
import pyfes
from pyproj import CRS
from coastsat import (
    SDS_download,
    SDS_preprocess,
    SDS_shoreline,
    SDS_tools,
    SDS_transects,
    SDS_slope,
)

def load_settings(config_path):
    """
    Load CLI-generated settings.json and resolve all relative paths, including FES config.
    """
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, 'r') as f:
        config = json.load(f)
    base_dir = config_path.parent

    inputs_config = config.get('inputs', {})
    # standard paths
    for key in ('aoi_path', 'reference_shoreline', 'transects'):
        if key in inputs_config:
            inputs_config[key] = str((base_dir / inputs_config[key]).resolve())
    # FES2022 YAML: absolute path
    if 'fes_config' in inputs_config:
        inputs_config['fes_config'] = str(Path(inputs_config['fes_config']).expanduser().resolve())
    
    if 'output_epsg' not in config:
        raise KeyError("settings.json must include an 'output_epsg' entry.")

    # output dir
    if 'output_dir' in config:
        config['output_dir'] = str((base_dir / config['output_dir']).resolve())

    return config


def initial_settings(config):
    """
    Initial settings for the shoreline extraction.
    Returns the inputs dictionary and settings dictionary.
    """
    # TODO: Ensure correct GeoJson
    # Load the polygon from a GeoJSON file)
    polygon = SDS_tools.polygon_from_kml(config['inputs']['aoi_path'])
    polygon = SDS_tools.smallest_rectangle(polygon)

    # Date range
    dates = ['1984-01-01', '2025-01-01']

    # TODO: Set sat list and date range
    # Satellites
    sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']
    # Name of the site
    sitename = config['inputs']['sitename']  

    # Filepath where data will be stored
    filepath_data = config['output_dir']  

    # Put all the inputs into a dictionary
    inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data,
        'reference_geojson': config['inputs']['reference_shoreline'],
        'transect_geojson':   config['inputs']['transects'],
        'fes_config': config['inputs']['fes_config'],
        # 'excluded_epsg_codes': ['32609'],
        # 'LandsatWRS': '055022',
        #'S2tile': '09UVA',
        # 'months': [7, 8, 9, 10],
        # 'skip_L7_SLC': True
    }

    # Before downloading the images, check how many images are available for your inputs
    # SDS_download.check_images_available(inputs)

    # Retrieve satellite images from GEE
    metadata = SDS_download.retrieve_images(inputs)
    metadata = SDS_download.get_metadata(inputs)

    # Settings for the shoreline extraction
    settings = {
        # General parameters:
        'cloud_thresh': 0.04,  # Threshold on maximum cloud cover
        'dist_clouds': 25,  # Distance around clouds where shoreline can't be mapped
        'output_epsg': config['output_epsg'],  # EPSG code of spatial reference system desired for the output
        # Quality control:
        'check_detection': False,  # If True, shows each shoreline detection to the user for validation
        'adjust_detection': False,  # If True, allows user to adjust the position of each shoreline by changing the threshold
        'save_figure': True,  # If True, saves a figure showing the mapped shoreline for each image
        # [ONLY FOR ADVANCED USERS] Shoreline detection parameters:
        'min_beach_area': 100,  # Minimum area (in metres^2) for an object to be labelled as a beach
        'min_length_sl': 300,  # Minimum length (in metres) of shoreline perimeter to be valid
        'cloud_mask_issue': False,  # Switch this parameter to True if sand pixels are masked (in black) on many images
        'sand_color': 'default',  # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'pan_off': False,  # True to switch pansharpening off for Landsat 7/8/9 imagery
        's2cloudless_prob': 20,  # Threshold to identify cloud pixels in the s2cloudless probability mask
        # Add the inputs defined previously
        'inputs': inputs,
    }

    return inputs, settings, metadata


def retrieve_images(inputs):
    """
    Retrieve images from GEE and load metadata.
    """
    # Before downloading the images, check how many images are available for your inputs
    SDS_download.check_images_available(inputs)

    # Retrieve satellite images from GEE
    metadata = SDS_download.retrieve_images(inputs)

    # If you have already downloaded the images, just load the metadata file
    metadata = SDS_download.get_metadata(inputs)

    return metadata


def batch_shoreline_detection(metadata, settings, inputs):
    """
    Perform batch shoreline detection.
    Returns the output dictionary.
    """
    # Preprocess images (cloud masking, pansharpening/down-sampling)
    #SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)
    # create MP4 timelapse animation
    #print("[Step 4] Generating RGB time-lapse animation (this may take several minutes)...")
    fn_animation = os.path.join(inputs['filepath'], '%s_animation_RGB.gif'%inputs['sitename'])
    fp_images = os.path.join(inputs['filepath'], 'jpg_files', 'preprocessed')
    fps = 10 # frames per second in animation
    #SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)
    try:
        filepath = settings['inputs']['filepath']
        with open(os.path.join(filepath, settings['inputs']['sitename'] + '_output' + '.pkl'), 'rb') as f:
            output = pickle.load(f)
            return output  # If the file exists, return `output` here and exit the function
    except FileNotFoundError:
        pass

    # Create a reference shoreline (helps to identify outliers and false detections)
    # settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
    settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_from_geojson(settings['inputs']['reference_geojson'],settings['output_epsg'])
    # Set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
    settings['max_dist_ref'] = 500

    # Extract shorelines from all images (also saves output.pkl and shorelines.kml)
    output = SDS_shoreline.extract_shorelines(metadata, settings)

    # Remove duplicates (images taken on the same date by the same satellite)
    output = SDS_tools.remove_duplicates(output)
    # Remove inaccurate georeferencing (set threshold to 10 m)
    output = SDS_tools.remove_inaccurate_georef(output, 10)

    # For GIS applications, save output into a GEOJSON layer
    geomtype = 'points'  # Choose 'points' or 'lines' for the layer geometry
    gdf = SDS_tools.output_to_gdf(output, geomtype)
    if gdf is None:
        raise Exception('Output does not contain any mapped shorelines')
    gdf.crs = CRS(settings['output_epsg'])  # Set layer projection
    # Save GEOJSON layer to file
    gdf.to_file(
        os.path.join(
            settings['inputs']['filepath'],
            '%s_output_%s.geojson' % (settings['inputs']['sitename'], geomtype),
        ),
        driver='GeoJSON',
        encoding='utf-8',
    )

    # create MP4 timelapse animation
    fn_animation = os.path.join(inputs['filepath'], '%s_animation_shorelines.gif'%inputs['sitename'])
    fp_images = os.path.join(inputs['filepath'], 'jpg_files', 'detection')
    fps = 10 # frames per second in animation
    SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)

    # Plot the mapped shorelines
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15,8], tight_layout=True)
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
        # plt.legend()
        fig.savefig(os.path.join(settings['inputs']['filepath'], 'mapped_shorelines.jpg'), dpi=200)
        plt.close(fig)
        # Optionally, display the plot
        # plt.show()

    return output


def shoreline_analysis(output, settings):
    """
    Analyze the shorelines and compute cross-shore distances along transects.
    Returns the cross_distance dictionary and transects dictionary.
    """
    # If you have already mapped the shorelines, load the output.pkl file
    # filepath = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'])
    # with open(os.path.join(filepath, settings['inputs']['sitename'] + '_output' + '.pkl'), 'rb') as f:
    #     output = pickle.load(f)
    # Remove duplicates (images taken on the same date by the same satellite)
    output = SDS_tools.remove_duplicates(output)
    # Remove inaccurate georeferencing (set threshold to 10 m)
    output = SDS_tools.remove_inaccurate_georef(output, 10)

    # Now we have to define cross-shore transects over which to quantify the shoreline changes
    # Each transect is defined by two points, its origin and a second point that defines its orientation
    
    print("[Step 5] Running shoreline change analysis for each transect...")

    # TODO: Set transects to be drawn
    # Option 1: Draw origin of transect first and then a second point to define the orientation
    # transects = SDS_transects.draw_transects(output, settings)

    # Option 2: Retrieve transects from geojson file
    # C:\Users\psteeves\coastal\planetscope_coastsat\user_inputs\transects
    # transects = SDS_tools.transects_from_geojson()
    transects = SDS_tools.transects_from_geojson(settings['inputs']['transect_geojson'])

    # Plot the transects to make sure they are correct (origin landwards!)
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))
        for i, key in enumerate(list(transects.keys())):
            plt.plot(transects[key][0, 0], transects[key][0, 1], 'bo', ms=5)
            plt.plot(transects[key][:, 0], transects[key][:, 1], 'k-', lw=1)
            if i % 10 == 0:
                plt.text(
                    transects[key][0, 0] - 100,
                    transects[key][0, 1] + 100,
                    key,
                    va='center',
                    ha='right',
                    bbox=dict(boxstyle='square', ec='k', fc='w'),
                )
        fig.savefig(
            os.path.join(settings['inputs']['filepath'], 'mapped_shorelines_with_transects.jpg'),
            dpi=200,
        )
        plt.close(fig)
        # Optionally, display the plot
        # plt.show()

    # Compute intersections with quality-control parameters
    settings_transects = {  # Parameters for computing intersections
        'along_dist': 25,  # Along-shore distance to use for computing the intersection
        'min_points': 3,  # Minimum number of shoreline points to calculate an intersection
        'max_std': 15,  # Max std for points around transect
        'max_range': 30,  # Max range for points around transect
        'min_chainage': -100,  # Largest negative value along transect (landwards of transect origin)
        'multiple_inter': 'max',  # Mode for removing outliers ('auto', 'nan', 'max')
        'auto_prc': 0.1,  # Percentage to use in 'auto' mode to switch from 'nan' to 'max'
    }

    
    cross_distance = SDS_transects.compute_intersection_QC(
        output, transects, settings_transects
    )



    # Plot the time-series of cross-shore shoreline change
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        gs = gridspec.GridSpec(len(cross_distance), 1)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
        for i, key in enumerate(cross_distance.keys()):
            if np.all(np.isnan(cross_distance[key])):
                continue
            ax = fig.add_subplot(gs[i, 0])
            ax.grid(linestyle=':', color='0.5')
            ax.set_ylim([-50, 50])
            ax.plot(
                output['dates'],
                cross_distance[key] - np.nanmedian(cross_distance[key]),
                '-o',
                ms=4,
                mfc='w',
            )
            ax.set_ylabel('distance [m]', fontsize=12)
            ax.text(
                0.5,
                0.95,
                key,
                bbox=dict(boxstyle='square', ec='k', fc='w'),
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=14,
            )
        fig.savefig(
            os.path.join(settings['inputs']['filepath'], 'time_series_raw.jpg'), dpi=200
        )
        plt.close(fig)
        # Optionally, display the plot
        # plt.show()

    # Save time-series in a .csv file
    out_dict = dict([])
    out_dict['dates'] = output['dates']
    for key in transects.keys():
        out_dict['Transect ' + key] = cross_distance[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(
        settings['inputs']['filepath'],
        'transect_time_series.csv',
    )
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s' % fn)
    dates = output['dates']
    print(f'Length of dates at end of shoreline_analysis: {len(dates)}')

    return cross_distance, transects, output


def tidal_correction(output, cross_distance, transects, settings, slope_est, dates_sat, tides_sat):
    """Perform tidal correction along each transect using specific slopes."""
    reference_elevation = 0
    cross_distance_tidally_corrected = {}

    for key in cross_distance.keys():
        common_length = min(len(dates_sat), len(tides_sat), len(cross_distance[key]))
        dates_sat = dates_sat[:common_length]
        tides_sat = tides_sat[:common_length]
        # Align cross_distance[key] to common_length
        cross_distance[key] = cross_distance[key][:common_length]

        # Perform tidal correction
        transect_slope = slope_est[key]  # Retrieve the specific slope for each transect
        correction = (tides_sat - reference_elevation) / transect_slope

        cross_distance_tidally_corrected[key] = cross_distance[key] + correction

    # Save tidally-corrected time-series to CSV
    out_dict = {'dates': dates_sat}
    for key in cross_distance_tidally_corrected.keys():
        out_dict[key] = cross_distance_tidally_corrected[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'], 'transect_time_series_tidally_corrected.csv')
    df.to_csv(fn, sep=',')
    print(f'Tidally-corrected time-series saved as:\n{fn}')

    return cross_distance_tidally_corrected


def improved_transects_plot(output, transects, cross_distance_tidally_corrected, settings):
    """
    Create a plot with transects colored based on their shoreline change trend.
    """
    sitename = settings['inputs']['sitename']
    trend_min, trend_max = -30, 30  # Define trend range (m/year)
    num_intervals = 100  # Number of color intervals

    # Create colormap and normalization
    cmap = cm.get_cmap('RdBu_r', num_intervals)  # Red for erosion, blue for accretion
    norm = mcolors.Normalize(vmin=trend_min, vmax=trend_max)

    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    ax.set_title(f"Transects Colored by Shoreline Change Trend ({sitename})", fontsize=14)
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.axis('equal')
    ax.grid(linestyle=':', color='0.5')

    # Plot shorelines
    for i in range(len(output['shorelines'])):
        sl = output['shorelines'][i]
        date = output['dates'][i]
        ax.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%Y-%m-%d'), alpha=0.5)

    # Plot transects with color based on trend
    for key in transects.keys():
        trend, _ = SDS_transects.calculate_trend(output['dates'], cross_distance_tidally_corrected[key])
        color = cmap(norm(trend))
        ax.plot(transects[key][:, 0], transects[key][:, 1], '-', color=color, lw=2)
        ax.plot(transects[key][0, 0], transects[key][0, 1], 'bo', ms=5)  # Origin marker

    # Add colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
    cbar.set_label('Shoreline Change Trend (m/year)', fontsize=12)
    cbar.set_ticks(range(trend_min, trend_max + 1, 5))
    cbar.ax.tick_params(labelsize=10)

    # Save the improved plot
    output_path = os.path.join(settings['inputs']['filepath'], 'transects_colored_by_trend_updated.jpg')
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Improved plot saved to {output_path}")


def time_series_post_processing(transects, settings, cross_distance_tidally_corrected, output):
    """
    Post-process the time series data.
    """
    sitename = settings['inputs']['sitename']
    filename_output = os.path.join(os.getcwd(), 'data', sitename, f'{sitename}_output.pkl')
    # with open(filename_output, 'rb') as f:
    #     output = pickle.load(f)

    # Plot the mapped shorelines
    # if settings.get('save_figure', False):
    #     fig = plt.figure(figsize=[15, 8], tight_layout=True)
    #     plt.axis('equal')
    #     plt.xlabel('Eastings')
    #     plt.ylabel('Northings')
    #     plt.grid(linestyle=':', color='0.5')
    #     plt.title(f"{len(output['shorelines'])} shorelines mapped at {sitename} from 1984")
    #     for i in range(len(output['shorelines'])):
    #         sl = output['shorelines'][i]
    #         date = output['dates'][i]
    #         plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))
    #     for i, key in enumerate(list(transects.keys())):
    #         plt.plot(transects[key][0, 0], transects[key][0, 1], 'bo', ms=5)
    #         plt.plot(transects[key][:, 0], transects[key][:, 1], 'k-', lw=1)
    #         plt.text(
    #             transects[key][0, 0] - 100,
    #             transects[key][0, 1] + 100,
    #             key,
    #             va='center',
    #             ha='right',
    #             bbox=dict(boxstyle='square', ec='k', fc='w'),
    #         )

    if settings.get('save_figure', False):
        # Define the range for trend rates (m/year)
        trend_min = -30
        trend_max = 30
        num_intervals = 61

        # Create a colormap
        cmap = cm.get_cmap('coolwarm', num_intervals)
        norm = mcolors.Normalize(vmin=trend_min, vmax=trend_max)

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(15, 8), tight_layout=True)
        norm = mcolors.Normalize(vmin=trend_min, vmax=trend_max)
        cmap = cm.get_cmap('coolwarm', num_intervals)

        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            ax.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))

        for key in transects.keys():
            trend, _ = SDS_transects.calculate_trend(output['dates'], cross_distance_tidally_corrected[key])
            color = cmap(norm(trend))
            ax.plot(transects[key][:, 0], transects[key][:, 1], '-', color=color, lw=2)
            ax.plot(transects[key][0, 0], transects[key][0, 1], 'bo', ms=5)

        # Add colorbar
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Shoreline Change Trend (m/year)', fontsize=12)
        cbar.set_ticks(np.linspace(trend_min, trend_max, num_intervals))
        cbar.ax.tick_params(labelsize=10)

        # Save the figure
        fig.savefig(
            os.path.join(settings['inputs']['filepath'], 'transects_colored_by_trend.jpg'),
            dpi=200
        )
        plt.close(fig)
        # plt.show()

    # Load the tidally-corrected time-series
    filepath = os.path.join(
        settings['inputs']['filepath'],
        'transect_time_series_tidally_corrected.csv'
    )
    df = pd.read_csv(filepath, parse_dates=['dates'])
    dates = [_.to_pydatetime() for _ in df['dates']]
    cross_distance = cross_distance_tidally_corrected

    # Remove outliers
    # Plot Otsu thresholds for the mapped shorelines
    if settings.get('save_figure', False):
        fig, ax = plt.subplots(1, 1, figsize=[12, 5], tight_layout=True)
        ax.grid(which='major', ls=':', lw=0.5, c='0.5')
        ax.plot(output['dates'], output['MNDWI_threshold'], 'o-', mfc='w')
        ax.axhline(y=-1, ls='--', c='r', label='otsu_threshold limits')
        ax.axhline(y=0, ls='--', c='r')
        ax.set(
            title='Otsu thresholds on MNDWI for the %d shorelines mapped'
            % len(output['shorelines']),
            ylim=[-1.2, 0.2],
            ylabel='otsu threshold',
        )
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(settings['inputs']['filepath'], 'otsu_thresholds.jpg'))
        plt.close(fig)
        # Optionally, display the plot
        # plt.show()

    # Remove outliers in the time-series (despiking)
    settings_outliers = {
        'otsu_threshold': [
            -1,
            0,
        ],  # Min and max intensity threshold used for contouring the shoreline
        'max_cross_change': 100,  # Maximum cross-shore change observable between consecutive timesteps
        'plot_fig': True,  # Whether to plot the intermediate steps
    }
    cross_distance = SDS_transects.reject_outliers(
        cross_distance, output, settings_outliers
    )

    # Seasonal averaging
    # Compute seasonal averages along each transect
    season_colors = {'DJF': 'C3', 'MAM': 'C1', 'JJA': 'C2', 'SON': 'C0'}
    for key in cross_distance.keys():
        chainage = cross_distance[key]
        # Remove NaNs
        idx_nan = np.isnan(chainage)
        dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
        chainage = chainage[~idx_nan]

        # Compute shoreline seasonal averages (DJF, MAM, JJA, SON)
        dict_seas, dates_seas, chainage_seas, list_seas = SDS_transects.seasonal_average(
            dates_nonan, chainage
        )

        trend, y = SDS_transects.calculate_trend(dates_seas, chainage_seas)

        # Plot seasonal averages
        if settings.get('save_figure', False):
            fig, ax = plt.subplots(1, 1, figsize=[14, 4], tight_layout=True)
            ax.grid(which='major', linestyle=':', color='0.5')
            ax.set_title('Time-series at %s' % key, x=0, ha='left')
            ax.set(ylabel='distance [m]')
            ax.plot(
                dates_nonan,
                chainage,
                '+',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                alpha=0.5,
                label='raw datapoints',
            )
            ax.plot(
                dates_seas,
                chainage_seas,
                '-',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                label='seasonally-averaged',
            )
            for k, seas in enumerate(dict_seas.keys()):
                ax.plot(
                    dict_seas[seas]['dates'],
                    dict_seas[seas]['chainages'],
                    'o',
                    mec='k',
                    color=season_colors[seas],
                    label=seas,
                    ms=5,
                )
            ax.plot(dates_seas,y,'--',color='b', label='trend %.1f m/year'%trend)
            ax.legend(
                loc='lower left',
                ncol=6,
                markerscale=1.5,
                frameon=True,
                edgecolor='k',
                columnspacing=1,
            )
            fig.savefig(os.path.join(settings['inputs']['filepath'], f'{key}_seasonal_average.jpg'))
            plt.close(fig)
            # Optionally, display the plot
            # plt.show()

    # Monthly averaging
    # Compute monthly averages along each transect
    month_colors = plt.get_cmap('tab20')
    for key in cross_distance.keys():
        chainage = cross_distance[key]
        # Remove NaNs
        idx_nan = np.isnan(chainage)
        dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
        chainage = chainage[~idx_nan]

        # Compute shoreline monthly averages
        dict_month, dates_month, chainage_month, list_month = SDS_transects.monthly_average(
            dates_nonan, chainage
        )

        # Plot monthly averages
        if settings.get('save_figure', False):
            fig, ax = plt.subplots(1, 1, figsize=[14, 4], tight_layout=True)
            ax.grid(which='major', linestyle=':', color='0.5')
            ax.set_title('Time-series at %s' % key, x=0, ha='left')
            ax.set(ylabel='distance [m]')
            ax.plot(
                dates_nonan,
                chainage,
                '+',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                alpha=0.5,
                label='raw datapoints',
            )
            ax.plot(
                dates_month,
                chainage_month,
                '-',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                label='monthly-averaged',
            )
            for k, month in enumerate(dict_month.keys()):
                ax.plot(
                    dict_month[month]['dates'],
                    dict_month[month]['chainages'],
                    'o',
                    mec='k',
                    color=month_colors(k),
                    label=month,
                    ms=5,
                )
            ax.legend(
                loc='lower left',
                ncol=7,
                markerscale=1.5,
                frameon=True,
                edgecolor='k',
                columnspacing=1,
            )
            fig.savefig(os.path.join(settings['inputs']['filepath'], f'{key}_monthly_average.jpg'))
            plt.close(fig)
            # Optionally, display the plot
            # plt.show()
    return cross_distance


def slope_estimation(settings, cross_distance, output):
    """
    Estimate the beach slope along each transect, while graphing the full tide series and then 
    filtering the acquisition dates for slope estimation.
    """
    print("[Step 6] Estimating beach slopes...")
    # Setup for output directory
    fp_slopes = os.path.join(settings['inputs']['filepath'], 'slope_estimation')
    if not os.path.exists(fp_slopes):
        os.makedirs(fp_slopes)
    print(f'Outputs will be saved in {fp_slopes}')

    if 'S2' in output['satname']:
        idx_S2 = np.array([_ == 'S2' for _ in output['satname']])
        for key in output.keys():
            output[key] = [output[key][_] for _ in np.where(~idx_S2)[0]]

    output = SDS_tools.remove_duplicates(output)
    # remove inaccurate georeferencing (set threshold to 10 m)
    output = SDS_tools.remove_inaccurate_georef(output, 10)

   
    transects = SDS_tools.transects_from_geojson(settings['inputs']['transect_geojson'])

    # compute intersections
    settings_transects = { # parameters for computing intersections
                        'along_dist':          25,        # along-shore distance to use for computing the intersection
                        'min_points':          3,         # minimum number of shoreline points to calculate an intersection
                        'max_std':             15,        # max std for points around transect
                        'max_range':           30,        # max range for points around transect
                        'min_chainage':        -100,      # largest negative value along transect (landwards of transect origin)
                        'multiple_inter':      'auto',    # mode for removing outliers ('auto', 'nan', 'max')
                        'auto_prc':            0.1,       # percentage of the time that multiple intersects are present to use the max
                        }
    cross_distance = SDS_transects.compute_intersection_QC(output, transects, settings_transects) 
    # remove outliers in the time-series (coastal despiking)
    settings_outliers = {'max_cross_change':   40,             # maximum cross-shore change observable between consecutive timesteps
                        'otsu_threshold':     [-.5,0],        # min and max intensity threshold use for contouring the shoreline
                        'plot_fig':           False,           # whether to plot the intermediate steps
                        }
    #cross_distance = SDS_transects.reject_outliers(cross_distance,output,settings_outliers)
    



    # Load FES2022 configuration for tide calculation
    print("Loading FES2022 config file...")
    fes_yaml = settings['inputs']['fes_config']
    handlers = pyfes.load_config(fes_yaml)
    ocean_tide = handlers['tide']
    load_tide = handlers['radial']

    aoi_geom = Polygon(settings['inputs']['polygon'][0])
    
    # Calculate tides at centroid
    centroid = SDS_tools.select_valid_centroid(aoi_geom, ocean_tide, load_tide)
    #centroid = [236.51805409, 48.65420802439958]

    # Generate full time-series tide data for graphing
    date_range = [
        pytz.utc.localize(datetime(1984, 1, 1)),
        pytz.utc.localize(datetime(2025, 1, 1)),
    ]
    timestep = 900  # seconds
    dates_ts, tides_ts = SDS_slope.compute_tide(
        centroid, date_range, timestep, ocean_tide, load_tide
    )


    # Retrieve tide levels at each satellite acquisition date
    dates_sat = output['dates']
    tides_sat = SDS_slope.compute_tide_dates(
        centroid, dates_sat, ocean_tide, load_tide
    )

    del ocean_tide, load_tide
    gc.collect()

    # Plotting the full tide series and satellite acquisition points
    if settings.get('save_figure', False):
        fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
        ax.grid(which='major', linestyle=':', color='0.5')
        ax.plot(dates_ts, tides_ts, '-', color='0.6', label='Full Tide Series')
        ax.plot(
            dates_sat,
            tides_sat,
            '-o',
            color='k',
            ms=6,
            mfc='w',
            lw=1,
            label='Image Acquisition',
        )
        ax.set(
            ylabel='Tide Level [m]',
            xlim=[dates_sat[0], dates_sat[-1]],
            title='Tide Levels at the Time of Image Acquisition',
        )
        ax.legend()
        fig.savefig(os.path.join(settings['inputs']['filepath'], '%s_tide_timeseries.jpg' % settings['inputs']['sitename']), dpi=200)
        plt.close(fig)

    # Settings for slope estimation
    settings_slope = {
        'slope_min': 0.005,
        'slope_max': 0.4,
        'delta_slope': 0.005,
        'n0': 50,
        'freq_cutoff': 1. / (24 * 3600 * 30),  # 30-day frequency
        'delta_f': 100 * 1e-10,
        'prc_conf': 0.05,
        'plot_fig': True,
    }

    beach_slopes = SDS_slope.range_slopes(settings_slope['slope_min'], settings_slope['slope_max'], settings_slope['delta_slope'])

    # Define the date range for slope estimation and filter `dates_sat` and `tides_sat`
    settings_slope['date_range'] = [2020, 2024]
    date_start = pytz.utc.localize(datetime(2020, 1, 1))
    date_end = pytz.utc.localize(datetime(2024, 1, 1))

    # Filter the `dates_sat` and `tides_sat` based on date range for slope estimation
    idx_dates = [date_start < date < date_end for date in dates_sat]
    selected_indices = np.where(idx_dates)[0]
    filtered_dates_sat = [dates_sat[i] for i in selected_indices]
    filtered_tides_sat = tides_sat[selected_indices]

    # Apply the same filtering to each key in cross_distance to ensure lengths match
    filtered_cross_distance = {}
    for key in cross_distance.keys():
        filtered_cross_distance[key] = cross_distance[key][selected_indices]

    # Plot the distribution of time steps for filtered data
    SDS_slope.plot_timestep(filtered_dates_sat)
    fig = plt.gcf()
    fig.savefig(os.path.join(fp_slopes, '0_timestep_distribution.jpg'), dpi=200)
    plt.close(fig)

    # Frequency settings and calculations
    settings_slope['n_days'] = 8
    settings_slope['freqs_max'] = SDS_slope.find_tide_peak(filtered_dates_sat, filtered_tides_sat, settings_slope)

    fig = plt.gcf()
    fig.savefig(os.path.join(fp_slopes, '1_tides_power_spectrum.jpg'), dpi=200)
    plt.close(fig)

    # Dictionary to store the slope estimates per transect
    print("[Step 6] Estimating beach slopes from elevation data...\n")
    slope_est, cis = dict([]), dict([])
    for key in cross_distance.keys():
        try:
            # Remove NaNs for the current transect
            idx_nan = np.isnan(filtered_cross_distance[key])
            dates = [filtered_dates_sat[i] for i in range(len(filtered_dates_sat)) if not idx_nan[i]]
            tide = np.array(filtered_tides_sat)[~idx_nan]
            composite = np.array(filtered_cross_distance[key])[~idx_nan]

            # Apply tidal correction and estimate slope for the current transect
            tsall = SDS_slope.tide_correct(composite, tide, beach_slopes)
            if len(dates) == 0 or len(tsall) == 0:
                print(f"Skipping transect {key} due to empty data.")
                slope_est[key], cis[key] = 0.1, (0.1, 0.1)
                continue
            slope_est[key], cis[key] = SDS_slope.integrate_power_spectrum(dates, tsall, settings_slope, key)
            fig = plt.gcf()
            fig.savefig(os.path.join(fp_slopes, f'2_energy_curve_{key}.jpg'), dpi=200)
            plt.close(fig)

            # Plot spectrum for each transect
            SDS_slope.plot_spectrum_all(dates, composite, tsall, settings_slope, slope_est[key])
            fig = plt.gcf()
            fig.savefig(os.path.join(fp_slopes, f'3_slope_spectrum_{key}.jpg'), dpi=200)
            plt.close(fig)
            print(f"  → {key}: Estimated slope = {slope_est[key]:.3f} m (CI: {cis[key][0]:.4f} – {cis[key][1]:.4f})")
        except Exception as e:
            print(f'Error processing transect {key}: {e}')
            print(f"Setting default slope for transect {key} to 0.1 due to error.")
            slope_est[key], cis[key] = 0.1, (0.1, 0.1)

    # Return slope estimates, filtered dates and tides, and the full dates/tides series
    return slope_est, dates_sat, tides_sat


def calculate_and_save_trends(transects, cross_distance_tidally_corrected, output, settings, slope_est):
    """
    Calculate the shoreline change trend for each transect and save it to a GeoJSON.
    """
    print("[Step 7] Estimating shoreline change trends for each transect...")
    trend_dict = {}
    for key in transects.keys():
        # Get valid distances and dates
        distances = cross_distance_tidally_corrected[key]
        valid_idx = ~np.isnan(distances)
        valid_dates = np.array(output['dates'])[valid_idx]
        valid_distances = distances[valid_idx]

        if len(valid_distances) > 1:
            trend, _ = SDS_transects.calculate_trend(valid_dates, valid_distances)
        else:
            trend = np.nan  # Not enough data to calculate trend

        trend_dict[key] = trend
        print(f"→ {key}: Trend = {trend:.4f} m/year")

    # Create a GeoDataFrame for transects
    transect_data = []
    for key, geometry in transects.items():

        seasonal_plot_filename = f'{key}_seasonal_average.jpg'
        seasonal_plot_path = os.path.join(settings['inputs']['filepath'], seasonal_plot_filename)

        slope_energy_curve_filename = f'2_energy_curve_{key}.jpg'
        slope_energy_curve_path = os.path.join(settings['inputs']['filepath'], 'slope_estimation', slope_energy_curve_filename)

        transect_data.append({
            'id': key,
            'geometry': MultiLineString([geometry]),
            'trend': trend_dict.get(key, np.nan),
            'slope': slope_est.get(key, 0.1),
            'plot_path': seasonal_plot_path,
            'slope_plot_path': slope_energy_curve_path
        })

    # Create GeoDataFrame
    gdf_transects = gpd.GeoDataFrame(transect_data, crs=CRS(settings['output_epsg']))

    # Save GeoJSON with trends
    geojson_path = os.path.join(
        settings['inputs']['filepath'], f"{settings['inputs']['sitename']}_transects_with_trends.geojson"
    )
    gdf_transects.to_file(geojson_path, driver='GeoJSON', encoding='utf-8')
    print(f"Transects with trends saved to {geojson_path}")

    return trend_dict


def main(config_path):
    
    config = load_settings(config_path)
    if not config:
        print("No valid configuration found. Exiting.")
        return
    
    print(f'Starting site: {config["inputs"]["sitename"]}')

    inputs, settings, metadata = initial_settings(config)
    # metadata = retrieve_images(inputs)
    output = batch_shoreline_detection(metadata, settings, inputs)
    cross_distance, transects, output = shoreline_analysis(output, settings)

    # Estimate slopes and retrieve tide data for tidal correction
    slope_est, dates_sat, tides_sat = slope_estimation(settings, cross_distance, output)
    cross_distance_tidally_corrected = tidal_correction(output, cross_distance, transects, settings, slope_est, dates_sat, tides_sat)
    trend_dict = calculate_and_save_trends(transects, cross_distance_tidally_corrected, output, settings, slope_est)
    improved_transects_plot(output, transects, cross_distance_tidally_corrected, settings)
    cross_distance = time_series_post_processing(transects, settings, cross_distance_tidally_corrected, output)

    print("Analysis completed successfully.")

    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run CoastSat analysis using a settings.json'
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to project settings.json'
    )
    args = parser.parse_args()
    main(args.config)
