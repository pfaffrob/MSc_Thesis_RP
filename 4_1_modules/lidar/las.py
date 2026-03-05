"""
LiDAR data processing module for reading and analyzing LAS point cloud files.
Provides LAS class for file I/O and point cloud manipulation, plus visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import gridspec
import os
import laspy
import numpy as np
from shapely.geometry import Polygon
import textwrap


class Hist:
    """Helper class for generating histograms of LAS file dimensions."""
    def __init__(self, las):
        self.las = las  # Already-read LAS object

    def __getattr__(self, name):
        name_lower = name.lower()
        if name_lower in self._dimension_names:
            def plot(bins=10):
                values = self.las[name_lower]

                if name_lower in ("x", "y", "z"):
                    values = np.round(values, decimals=0)

                unique_vals = np.unique(values)
                bins = min(bins, len(unique_vals))

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(values, bins=bins, edgecolor='black', color='#4c72b0', alpha=0.8)
                ax.set_title(f'Histogram of {name_lower} Dimension', fontsize=14, weight='bold')
                ax.set_xlabel(name_lower, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.ticklabel_format(useOffset=False, style='plain') # Disable scientific notation

                if bins < 10:
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
                    plt.xticks(rotation=90) # Rotate x-axis labels to vertical
                else:
                    ax.xaxis.set_major_formatter(mticker.NullFormatter()) # Hide x-axis labels
                    ax.xaxis.set_major_locator(mticker.NullLocator()) # Hide x-axis ticks

                plt.tight_layout()
                plt.show()


            return plot
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class LAS:
    """
    LAS point cloud file handler with utilities for reading, filtering, and analyzing LiDAR data.
    Provides access to point coordinates, classification, and various LAS dimensions.
    """
    classification_labels = {
        0: "Created, Never Classified", 1: "Unclassified", 2: "Ground", 3: "Low Vegetation",
        4: "Medium Vegetation", 5: "High Vegetation", 6: "Building", 7: "Low Point (Noise)",
        8: "Model Key Point (Mass Point)", 9: "Water", 10: "Rail", 11: "Road Surface",
        12: "Reserved", 13: "Wire - Guard", 14: "Wire - Conductor", 15: "Transmission Tower",
        16: "Wire-structure Connector", 17: "Bridge Deck", 18: "High Noise"
    }

    def __init__(self, file_path):
        """Initialize LAS reader with file path."""
        self.file_path = file_path
        self.file_name = os.path.basename(self.file_path)
        self.las_file = laspy.read(file_path)
        self.header = self.las_file.header
        self.hist = Hist(self.las_file)
        all_dimensions = self.las_file.point_format.dimensions
        self._actual_dims = {dim.name.lower(): dim for dim in all_dimensions}
        self._dimension_names = [
            dim_name.lower() for dim_name in self.las_file.point_format.dimension_names
            if dim_name.lower() in self._actual_dims
        ]

    @property
    def x(self):
        return self.las_file.x
    
    @property
    def y(self):
        return self.las_file.y
    
    @property
    def z(self):
        return self.las_file.z

    @property
    def points(self):
        return np.vstack((self.x, self.y, self.z)).T

    @property
    def crs(self):
        crs = self.header.parse_crs()
        return crs.to_epsg() if crs else None

    @property
    def extent(self):
        return {
            "min_x": np.min(self.x),
            "max_x": np.max(self.x),
            "min_y": np.min(self.y),
            "max_y": np.max(self.y)
        }

    @property
    def geom(self):
        ext = self.extent
        return Polygon([
            (ext["min_x"], ext["min_y"]),
            (ext["min_x"], ext["max_y"]),
            (ext["max_x"], ext["max_y"]),
            (ext["max_x"], ext["min_y"]),
            (ext["min_x"], ext["min_y"])
        ])
    
    @property
    def info(self):
        header_lines = []
        for attr in dir(self.header):
            if not attr.startswith("_") and not callable(getattr(self.header, attr)):
                line = f"{attr.replace('_', ' ').title()}: {getattr(self.header, attr)}"
                header_lines.extend(textwrap.wrap(line, width=120))

        if self.header.parse_crs() is not None:
            header_lines.append(str(self.header.parse_crs()))

        num_points_by_return = self.header.number_of_points_by_return
        non_zero_returns = np.trim_zeros(num_points_by_return, 'b')
        return_table = [[str(i), str(count)] for i, count in enumerate(non_zero_returns, start=1)]

        dim_table = []
        try:
            all_dimensions = self.las_file.point_format.dimensions
            actual_dims = {dim.name: dim for dim in all_dimensions}
            dimension_names = self.las_file.point_format.dimension_names

            for dim_name in dimension_names:
                if dim_name in actual_dims:
                    dim = actual_dims[dim_name]
                    dtype = dim.dtype
                    dtype_str = str(dtype) if dtype is not None else "bit field"
                    try:
                        values = getattr(self.las_file, dim_name.lower())
                        unique_vals = np.unique(values)
                        present = "Yes" if len(unique_vals) > 1 else "No"
                        stat = f"{np.min(values)} - {np.max(values)}" if np.issubdtype(values.dtype, np.number) else ", ".join(map(str, unique_vals))
                    except Exception:
                        present = "No"
                        stat = "N/A"
                    dim_table.append([dim_name, present, dtype_str, stat])
                else:
                    dim_table.append([dim_name, "No", "-", "-"])
        except Exception as e:
            dim_table.append([f"Could not list dimensions: {e}", "", "", ""])

        cls_table = []
        try:
            classifications = self.las_file.classification
            unique_classes, counts = np.unique(classifications, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                description = self.classification_labels.get(cls, "Unknown")
                cls_table.append([str(cls), str(count), description])
        except Exception as e:
            cls_table.append([f"Could not read classifications: {e}", "", ""])

        header_height = len(header_lines) * 0.2
        return_height = len(return_table) * 0.2
        dim_height = len(dim_table) * 0.2
        cls_height = len(cls_table) * 0.2
        total_height = header_height + return_height + dim_height + cls_height + 2

        fig = plt.figure(figsize=(14, total_height))
        gs = gridspec.GridSpec(4, 1, height_ratios=[header_height, return_height, dim_height, cls_height])
        fig.suptitle("LAS File Summary", fontsize=16, y=0.97)

        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        ax_header.text(0, 1, "\n".join(header_lines), fontsize=10, va='top', ha='left', family='monospace')
        ax_header.set_title("Header Information", loc='left', fontsize=12, weight='bold')

        ax_return = fig.add_subplot(gs[1])
        ax_return.axis('off')
        ax_return.set_title("Number of Points by Return", loc='left', fontsize=12, weight='bold')
        rtrn_table_fig = ax_return.table(
            cellText=return_table,
            colLabels=["Return Number", "Number of Points"],
            loc='upper center',
            cellLoc='left'
        )
        rtrn_table_fig.auto_set_font_size(False)
        rtrn_table_fig.set_fontsize(9)
        rtrn_table_fig.scale(1, 1.2)

        ax_dims = fig.add_subplot(gs[2])
        ax_dims.axis('off')
        ax_dims.set_title("Point Format Dimensions", loc='left', fontsize=12, weight='bold')
        dim_table_fig = ax_dims.table(
            cellText=dim_table,
            colLabels=["Dimension Name", "Present", "Data Type", "Value Summary"],
            loc='upper center',
            cellLoc='left'
        )
        dim_table_fig.auto_set_font_size(False)
        dim_table_fig.set_fontsize(9)
        dim_table_fig.scale(1, 1.2)

        ax_cls = fig.add_subplot(gs[3])
        ax_cls.axis('off')
        ax_cls.set_title("Point Classifications", loc='left', fontsize=12, weight='bold')
        cls_table_fig = ax_cls.table(
            cellText=cls_table,
            colLabels=["Class", "Count", "Description"],
            loc='upper center',
            cellLoc='left'
        )
        cls_table_fig.auto_set_font_size(False)
        cls_table_fig.set_fontsize(9)
        cls_table_fig.scale(1, 1.2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    @property
    def dims(self):
        for dim in self._dimension_names:
            print(dim)
    

    def to_file(self, output):
        """
        Write the LAS data to a file.

        Parameters:
            output_path (str): Either a full file path or a directory.
        """
        import os

        # If a directory is provided, append the original filename
        if os.path.isdir(output):
            output = os.path.join(output, self.file_name)

        self.las_file.write(output)


    @classmethod
    def _from_lasdata(cls, las_data, file_name=None):
        """
        Internal constructor to create a new LAS instance from a laspy.LasData object.

        Parameters:
            las_data (laspy.LasData): The LAS data to wrap.
            file_name (str, optional): Optional name to assign to the new instance.

        Returns:
            LAS: A new LAS instance.
        """
        instance = cls.__new__(cls)
        instance.las_path = None
        instance.file_name = file_name
        instance.las_file = las_data
        instance.header = las_data.header
        instance.hist = Hist(las_data)

        all_dimensions = las_data.point_format.dimensions
        instance._actual_dims = {dim.name.lower(): dim for dim in all_dimensions}
        instance._dimension_names = [
            dim_name.lower() for dim_name in las_data.point_format.dimension_names
            if dim_name.lower() in instance._actual_dims
        ]
        return instance

    
    def clip(self, extent, prepend_file_name = '_clipped'):
        """
        Clip the LAS data to a specified extent.

        Parameters:
        - extent: A tuple (xmin, xmax, ymin, ymax[, zmin, zmax])
        Returns:
        - A new LAS instance with clipped data.
        """
        x, y = self.las_file.x, self.las_file.y
        mask = (x >= extent[0]) & (x <= extent[1]) & (y >= extent[2]) & (y <= extent[3])

        if len(extent) == 6:
            z = self.las_file.z
            mask &= (z >= extent[4]) & (z <= extent[5])
            
        new_las_data = laspy.LasData(self.las_file.header)
        new_las_data.points = self.las_file.points[mask]

        
        # Prepend '_clipped' to the base filename
        base, ext = os.path.splitext(self.file_name)
        clipped_name = f"{base}{prepend_file_name}{ext}"

        return LAS._from_lasdata(new_las_data, file_name=clipped_name)

    def filter_by_classification(self, classes):
        """
        Returns a new LAS instance containing only points of the specified classification(s).

        Parameters:
            classes (int, str, or list): Classification code(s) or name(s), e.g., 2, 'Ground', or ['Ground', 6].

        Returns:
            LAS: A new LAS instance with filtered points.
        """
        # Normalize input to list
        if not isinstance(classes, (list, tuple, set)):
            classes = [classes]

        # Build name-to-code mapping
        label_to_code = {v.lower(): k for k, v in self.classification_labels.items()}

        # Resolve all class codes
        class_codes = set()
        for cls in classes:
            if isinstance(cls, int):
                class_codes.add(cls)
            elif isinstance(cls, str):
                cls_lower = cls.lower()
                if cls_lower not in label_to_code:
                    raise ValueError(f"Unknown classification name: {cls}")
                class_codes.add(label_to_code[cls_lower])
            else:
                raise TypeError(f"Unsupported classification type: {type(cls)}")

        # Filter points
        mask = np.isin(self.las_file.classification, list(class_codes))

        new_las_data = laspy.LasData(self.las_file.header)
        new_las_data.points = self.las_file.points[mask]

        base, ext = os.path.splitext(self.file_name)
        file_name = f"{base}_filtered{ext}"

        return LAS._from_lasdata(new_las_data, file_name=file_name)

    def filter_by_return(self, returns):
        """
        Returns a new LAS instance containing only points with the specified return number(s).

        Parameters:
            returns (int or list of int): Return number(s) to filter by (e.g., 1 for first return).

        Returns:
            LAS: A new LAS instance with filtered points.
        """
        # Normalize input to list
        if not isinstance(returns, (list, tuple, set)):
            returns = [returns]

        # Validate return numbers
        if not all(isinstance(r, int) and r > 0 for r in returns):
            raise ValueError("Return numbers must be positive integers.")

        # Filter points
        mask = np.isin(self.las_file.return_number, returns)

        new_las_data = laspy.LasData(self.las_file.header)
        new_las_data.points = self.las_file.points[mask]

        base, ext = os.path.splitext(self.file_name)
        file_name = f"{base}_filtered{ext}"

        return LAS._from_lasdata(new_las_data, file_name=file_name)
    
    def to_dem(self, output, type, resolution, fill_gaps):
        """
        Export a DEM GeoTIFF from this LAS object.

        Parameters:
            output_path (str): Either a full file path or a directory.
            type (str): 'DTM' for ground-only or 'DSM' for first returns.
            grid_resolution (float): Grid resolution for the DEM.
            fill_missing (bool): Whether to fill missing values in the DEM.
        """
        import os

        # Validate type
        if type not in ('DTM', 'DSM'):
            raise ValueError("type must be either 'DTM' or 'DSM'")

        # Determine output directory based on resolution
        res_str = f"{int(resolution * 100):03d}"
        res_folder = f"res_{res_str}"
        output_dir = os.path.join(output, res_folder)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct the output file path
        base_name = os.path.splitext(self.file_name)[0]
        file_name = f"{base_name}_{type}.tif"
        output = os.path.join(output_dir, file_name)


        # Filter LAS data
        if type == 'DTM':
            las = self.filter_by_classification(2)
        elif type == 'DSM':
            las = self.filter_by_return(1)

        save_dem_to_tif(las, output, resolution, fill_gaps)

    def subtract_raster(self, raster_path):
        """
        Subtracts raster elevation values from LAS Z values for points within the raster extent
        and with valid (non-nodata) raster values.

        Parameters:
            raster_path (str): Path to the raster (.tif) file.

        Returns:
            LAS: A new LAS instance with updated Z values, excluding points with invalid raster values.
        """
        import rasterio
        from rasterio.transform import rowcol
        import numpy as np

        with rasterio.open(raster_path) as src:
            raster = src.read(1)
            transform = src.transform
            nodata = src.nodata

            x = self.x
            y = self.y
            z = self.z

            rows, cols = rowcol(transform, x, y)

            # Check bounds
            in_bounds = (
                (rows >= 0) & (rows < raster.shape[0]) &
                (cols >= 0) & (cols < raster.shape[1])
            )

            # Filter to in-bounds points
            valid_rows = rows[in_bounds]
            valid_cols = cols[in_bounds]
            valid_indices = np.where(in_bounds)[0]

            raster_values = raster[valid_rows, valid_cols]

            # Check for nodata
            if nodata is not None:
                not_nodata = raster_values != nodata
            else:
                not_nodata = ~np.isnan(raster_values)

            # Final valid indices
            final_valid_indices = valid_indices[not_nodata]
            final_raster_values = raster_values[not_nodata]

            # Subtract raster from Z
            new_z = z[final_valid_indices] - final_raster_values

            # Ensure Z values are finite
            finite_mask = np.isfinite(new_z)
            final_valid_indices = final_valid_indices[finite_mask]
            new_z = new_z[finite_mask]

            # Create new LAS object
            new_las_data = laspy.LasData(self.las_file.header)
            new_las_data.points = self.las_file.points[final_valid_indices]
            new_las_data.z = new_z

        return self._from_lasdata(new_las_data, self.file_name)


    # def dim_mask(self, dim_name):
    #     if dim_name not in self._dimension_names:
    #         raise AttributeError(f"{dim_name} is not available in the extra dimensions.")
    #     return self.las_file[dim_name]



    def filter_by_dim(self, dim_name, out_dir, num_threshold):
        """
        Filters LAS points based on the 'final_segs' extra byte field.
        Keeps only classes with >= 100000 points.
        Saves the filtered LAS file to the given output directory.
        """
        import os

        # Access final_segs safely
        try:
            dim_mask = self.las_file[dim_name]
        except AttributeError as e:
            raise ValueError(f"Extra byte field {dim_name} not found.") from e

        # Count points per class
        unique_classes, counts = np.unique(dim_mask, return_counts=True)
        valid_classes = [cls for cls, count in zip(unique_classes, counts) if count >= num_threshold]

        # Filter mask
        mask = np.isin(dim_mask, valid_classes)
        filtered_points = self.las_file.points[mask]

        # Create new LAS object
        new_las = laspy.LasData(self.header)
        new_las.points = filtered_points

        # Save to output directory
        out_path = os.path.join(out_dir, self.file_name)
        new_las.write(out_path)










def generate_dem(las, grid_resolution):
    """
    Generate a DEM from a filtered LAS object using median elevation values.

    Parameters:
        las (laspy.LasData): LAS object with point cloud data (already filtered).
        grid_resolution (float): Size of each raster cell.

    Returns:
        xi, yi, zi: Meshgrid coordinates and elevation grid with np.nan as NoData.
    """
    x, y, z = las.x, las.y, las.z

    # Define grid bounds
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    # Create grid indices
    xi = np.arange(xmin, xmax, grid_resolution)
    yi = np.arange(ymin, ymax, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Assign each point to a grid cell
    col_idx = ((x - xmin) // grid_resolution).astype(int)
    row_idx = ((y - ymin) // grid_resolution).astype(int)

    # Create a 2D array to hold the elevation values
    zi = np.full((len(yi), len(xi)), np.nan)

    # Group z-values by grid cell
    from collections import defaultdict
    cell_dict = defaultdict(list)
    for r, c, val in zip(row_idx, col_idx, z):
        if 0 <= r < len(yi) and 0 <= c < len(xi):
            cell_dict[(r, c)].append(val)

    # Fill the grid with the median value
    for (r, c), values in cell_dict.items():
        if values:
            zi[r, c] = np.median(values)

    return xi_grid, yi_grid, zi


def save_dem_to_tif(las, output_path, grid_resolution, fill_gaps):
    """
    Save a DEM grid to a GeoTIFF file with CRS 2193 and LZW compression.
    Optionally fills missing data using a mean filter of specified window size.
    
    Parameters:
        las: LAS object
        output_path (str): Path to output .tif file
        grid_resolution (float): Grid resolution in map units
        fill_gaps (int or False): If int, applies a mean filter of that size to fill NaNs.
                                  If False, no gap filling is performed.
    """
    import rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import generic_filter
    import warnings

    xi, yi, zi = generate_dem(las, grid_resolution)

    pixel_size_x = xi[0, 1] - xi[0, 0]
    pixel_size_y = yi[1, 0] - yi[0, 0]

    origin_x = np.floor(xi.min() / grid_resolution) * grid_resolution
    origin_y = np.ceil(yi.max() / grid_resolution) * grid_resolution

    transform = from_origin(origin_x, origin_y, pixel_size_x, pixel_size_y)

    zi_flipped = np.flipud(zi)

    # Fill gaps if fill_gaps is an integer
    if isinstance(fill_gaps, int) and fill_gaps > 1:
        def nanmean_filter(values):
            return np.nanmean(values)

        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            filled = generic_filter(zi_flipped, nanmean_filter, size=fill_gaps, mode='mirror')
        zi_flipped = np.where(np.isnan(zi_flipped), filled, zi_flipped)

    profile = {
        'driver': 'GTiff',
        'height': zi_flipped.shape[0],
        'width': zi_flipped.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:2193',
        'transform': transform,
        'compress': 'LZW',
        'nodata': np.nan
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(zi_flipped.astype('float32'), 1)
