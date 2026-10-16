+*In[ ]:*+
[source, ipython3]
----
# =============================================================================
# LSR geological-boundary estimation workflow
# =============================================================================
# Author: Thomas Yeo
# Created: 23 July 2026
# Updated: 16 October 2026
# Version: 1.1
#
# Description:
# This script was developed for estimating a geological boundary 
# using LSR. The input data consist of X, Y, and Z coordinates
# of points interpreted to lie on a geological boundary.
#
# The script fits a plane using orthogonal least squares, calculate
# point-to-plane residual distances, determines the planar orientation,
# identifies and removes potential outliers using a MAD-based residual
# filter, refits the plane using the filtered dataset, and exports
# the resulting data, plane parameters, and summary statistics.
# ============================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set global plotting style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,
    'font.weight': 'bold'
})

# ========================
# 1. User Settings
# ========================
# Input text file containing only X, Y, Z coordinates with no heading
# The file should be space- or tab-separated
input_file = "/Users/YourUserName/FileLocation/FileName.txt" #<-- change this to your file location

output_dir = "/Users/YourUserName/FileLocation"              #<-- change this to your save folder
os.makedirs(output_dir, exist_ok=True)

# Coordinate convention:
# x = Easting
# y = Northing
# z = Elevation
x_col = "x"  #<-- change this to your coordinate convention
y_col = "y"  #<-- change this to your coordinate convention
z_col = "z"  #<-- change this to your coordinate convention

# Outlier removal setting
# Median Absolute Deviation-based residual filter
# Points with modified Z-scores greater than this value are removed
mad_threshold = 3.5 #Iglewicz and Hoaglin (1993)

# Start Timer
start_time = time.perf_counter()

# ========================
# 2. Functions
# ========================
def load_boundary_points(input_file):
    """
    Loads boundary points from a text file with no heading.
    
    Input format:
        x y z
        
    The file must contain space- or tab-separated coordinates.
    """

    data = pd.read_csv(
        input_file,
        sep=r"\s+",
        header=None,
        names=["x", "y", "z"]
    )

    data = data.dropna().reset_index(drop=True)
    data["point_id"] = np.arange(len(data))

    return data


def fit_lsr_plane(data):
    """
    Fits a best-fit plane using orthogonal LSR.

    Plane equation:
        normal_x * x + normal_y * y + normal_z * z + intercept = 0

    The plane is obtained using singular value decomposition.
    The plane normal corresponds to the direction of minimum variance.
    """

    points = data[["x", "y", "z"]].values

    centroid = points.mean(axis=0)
    centered_points = points - centroid

    _, _, vh = np.linalg.svd(centered_points)

    normal_vector = vh[-1, :]

    # Keep a consistent upward normal direction
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    intercept = -np.dot(normal_vector, centroid)

    return normal_vector, intercept, centroid


def calculate_point_to_plane_residuals(data, normal_vector, intercept):
    """
    Calculates signed and absolute orthogonal residuals from each point
    to the fitted plane.

    Residual equation:
        r_i = (normal_x*x_i + normal_y*y_i + normal_z*z_i + intercept)
              / sqrt(normal_x^2 + normal_y^2 + normal_z^2)
    """

    points = data[["x", "y", "z"]].values
    norm_w = np.linalg.norm(normal_vector)

    signed_residuals = (
        points @ normal_vector + intercept
    ) / norm_w

    absolute_residuals = np.abs(signed_residuals)

    return signed_residuals, absolute_residuals


def compute_plane_orientation(normal_vector):
    """
    Computes dip and dip direction from the plane normal vector.

    Coordinate convention:
    x = Easting
    y = Northing
    z = Elevation

    Geological convention:
    Dip direction azimuth is measured clockwise from north.
    """

    a, b, c = normal_vector

    horizontal_length = np.sqrt(a**2 + b**2)

    dip = np.degrees(
        np.arctan2(horizontal_length, abs(c))
    )

    z_axis = np.array([0.0, 0.0, 1.0])
    strike = np.cross(normal_vector, z_axis)

    if np.linalg.norm(strike) == 0:
        dip_direction = np.nan
        dip_vec = np.array([np.nan, np.nan, np.nan])
    else:
        dip_vec = np.cross(normal_vector, strike)
        dip_vec = dip_vec / np.linalg.norm(dip_vec)

        # Ensure the dip vector points downward
        if dip_vec[2] > 0:
            dip_vec = -dip_vec

        dip_direction = (
            np.degrees(np.arctan2(dip_vec[0], dip_vec[1])) + 360
        ) % 360

    return dip, dip_direction, dip_vec


def calculate_residual_statistics(signed_residuals, absolute_residuals):
    """
    Calculates residual error statistics for the fitted plane.
    """

    statistics = {
        "mean_signed_residual": np.mean(signed_residuals),
        "mean_absolute_residual": np.mean(absolute_residuals),
        "median_absolute_residual": np.median(absolute_residuals),
        "rmse": np.sqrt(np.mean(signed_residuals**2)),
        "std_signed_residual": np.std(signed_residuals),
        "min_signed_residual": np.min(signed_residuals),
        "max_signed_residual": np.max(signed_residuals),
        "max_absolute_residual": np.max(absolute_residuals)
    }

    return statistics


def remove_outliers_by_mad(data, absolute_residuals, mad_threshold):
    """
    Removes potential outlier points using a MAD filter applied to
    the absolute point-to-plane residuals.

    This method is recommended because it is robust against extreme
    residual values and does not assume normally distributed errors.
    """

    median_residual = np.median(absolute_residuals)
    mad = np.median(np.abs(absolute_residuals - median_residual))

    if mad == 0:
        data["modified_z_score"] = 0.0
        data["outlier"] = False
        return data.copy(), data.iloc[0:0].copy()

    modified_z_score = (
        0.6745 * (absolute_residuals - median_residual) / mad
    )

    data = data.copy()
    data["modified_z_score"] = modified_z_score
    data["outlier"] = np.abs(modified_z_score) > mad_threshold

    cleaned_data = data[data["outlier"] == False].copy()
    outlier_data = data[data["outlier"] == True].copy()

    return cleaned_data, outlier_data


def build_plane_summary(
    model_name,
    normal_vector,
    intercept,
    dip,
    dip_direction,
    dip_vec,
    residual_statistics,
    number_of_points
):
    """
    Builds a summary table for the LSR plane.
    """

    summary = {
        "model": model_name,
        "normal_x": normal_vector[0],
        "normal_y": normal_vector[1],
        "normal_z": normal_vector[2],
        "intercept": intercept,
        "dip": dip,
        "dip_direction": dip_direction,
        "down_dip_x": dip_vec[0],
        "down_dip_y": dip_vec[1],
        "down_dip_z": dip_vec[2],
        "number_of_points": number_of_points
    }

    summary.update(residual_statistics)

    return summary


# ========================
# 3. Load Boundary Point Data
# ========================
boundary_data = load_boundary_points(input_file)

print("\nLoaded boundary points")
print("----------------------------")
print(f"Input file: {input_file}")
print(f"Number of points: {len(boundary_data)}")


# ========================
# 4. Initial LSR Plane Fitting
# ========================
normal_initial, intercept_initial, centroid_initial = fit_lsr_plane(
    boundary_data
)

signed_residuals_initial, absolute_residuals_initial = (
    calculate_point_to_plane_residuals(
        boundary_data,
        normal_initial,
        intercept_initial
    )
)

dip_initial, dip_direction_initial, dip_vec_initial = (
    compute_plane_orientation(normal_initial)
)

residual_stats_initial = calculate_residual_statistics(
    signed_residuals_initial,
    absolute_residuals_initial
)

boundary_data["signed_residual"] = signed_residuals_initial
boundary_data["absolute_residual"] = absolute_residuals_initial

initial_summary = build_plane_summary(
    model_name="Initial_LSR",
    normal_vector=normal_initial,
    intercept=intercept_initial,
    dip=dip_initial,
    dip_direction=dip_direction_initial,
    dip_vec=dip_vec_initial,
    residual_statistics=residual_stats_initial,
    number_of_points=len(boundary_data)
)

print("\nInitial LSR plane")
print("-----------------")
print(f"normal_x: {normal_initial[0]:.6f}")
print(f"normal_y: {normal_initial[1]:.6f}")
print(f"normal_z: {normal_initial[2]:.6f}")
print(f"intercept: {intercept_initial:.6f}")
print(f"Dip: {dip_initial:.2f}°")
print(f"Dip direction: {dip_direction_initial:.2f}°")
print(f"Mean absolute residual: {residual_stats_initial['mean_absolute_residual']:.4f} m")
print(f"RMSE: {residual_stats_initial['rmse']:.4f} m")


# ========================
# 5. Residual-Based Outlier Removal
# ========================
cleaned_data, outlier_data = remove_outliers_by_mad(
    boundary_data,
    absolute_residuals_initial,
    mad_threshold
)

print("\nOutlier removal")
print("---------------")
print(f"Method: MAD filter on absolute point-to-plane residuals")
print(f"MAD threshold: {mad_threshold}")
print(f"Number of retained points: {len(cleaned_data)}")
print(f"Number of removed outliers: {len(outlier_data)}")


# ========================
# 6. Refit LSR Plane After Outlier Removal
# ========================
normal_cleaned, intercept_cleaned, centroid_cleaned = fit_lsr_plane(
    cleaned_data
)

signed_residuals_cleaned, absolute_residuals_cleaned = (
    calculate_point_to_plane_residuals(
        cleaned_data,
        normal_cleaned,
        intercept_cleaned
    )
)

dip_cleaned, dip_direction_cleaned, dip_vec_cleaned = (
    compute_plane_orientation(normal_cleaned)
)

residual_stats_cleaned = calculate_residual_statistics(
    signed_residuals_cleaned,
    absolute_residuals_cleaned
)

cleaned_data["signed_residual_refitted"] = signed_residuals_cleaned
cleaned_data["absolute_residual_refitted"] = absolute_residuals_cleaned

cleaned_summary = build_plane_summary(
    model_name="Cleaned_LSR",
    normal_vector=normal_cleaned,
    intercept=intercept_cleaned,
    dip=dip_cleaned,
    dip_direction=dip_direction_cleaned,
    dip_vec=dip_vec_cleaned,
    residual_statistics=residual_stats_cleaned,
    number_of_points=len(cleaned_data)
)

print("\nCleaned LSR plane after outlier removal")
print("---------------------------------------")
print(f"normal_x: {normal_cleaned[0]:.6f}")
print(f"normal_y: {normal_cleaned[1]:.6f}")
print(f"normal_z: {normal_cleaned[2]:.6f}")
print(f"intercept: {intercept_cleaned:.6f}")
print(f"Dip: {dip_cleaned:.2f}°")
print(f"Dip direction: {dip_direction_cleaned:.2f}°")
print(f"Mean absolute residual: {residual_stats_cleaned['mean_absolute_residual']:.4f} m")
print(f"RMSE: {residual_stats_cleaned['rmse']:.4f} m")


# ========================
# 7. Export Results
# ========================
initial_points_path = os.path.join(
    output_dir,
    "LSR_initial_point_residuals.csv"
)

cleaned_points_path = os.path.join(
    output_dir,
    "LSR_cleaned_point_residuals.csv"
)

outlier_points_path = os.path.join(
    output_dir,
    "LSR_removed_outliers.csv"
)

summary_path = os.path.join(
    output_dir,
    "LSR_plane_summary.csv"
)

boundary_data.to_csv(initial_points_path, index=False)
cleaned_data.to_csv(cleaned_points_path, index=False)
outlier_data.to_csv(outlier_points_path, index=False)

summary_df = pd.DataFrame([
    initial_summary,
    cleaned_summary
])

summary_df["outlier_method"] = (
    f"MAD filter on absolute residuals, threshold = {mad_threshold}"
)

summary_df.to_csv(summary_path, index=False)


# ========================
# 8. Plot: Residual Distribution
# ========================
plt.figure(figsize=(9, 6))

plt.hist(
    boundary_data["absolute_residual"],
    bins=30,
    alpha=0.7,
    label="Initial residuals"
)

plt.hist(
    cleaned_data["absolute_residual_refitted"],
    bins=30,
    alpha=0.7,
    label="Cleaned residuals"
)

plt.xlabel("Absolute point-to-plane residual (m)")
plt.ylabel("Frequency")
plt.title("LSR Boundary Residual Distribution")
plt.legend()
plt.grid(True)

residual_histogram_path = os.path.join(
    output_dir,
    "LSR_residual_distribution.png"
)

plt.savefig(residual_histogram_path, dpi=600, bbox_inches="tight")
plt.show()

# ========================
# 9. Interactive 3D Plot: LSR Plane and Boundary Points
# ========================
# Select which plane to plot:
# Use cleaned LSR result if outliers were removed; otherwise use initial result.
normal_plot = normal_cleaned
intercept_plot = intercept_cleaned
plot_data = cleaned_data.copy()
plot_title = "Filtered geological boundary"

# If you want to plot the initial plane instead, use:
# normal_plot = normal_initial
# intercept_plot = intercept_initial
# plot_data = boundary_data.copy()
# plot_title = "Unfiltered geological boundary"

a, b, c = normal_plot

# Define plotting range from point coordinates
x_min, x_max = plot_data["x"].min(), plot_data["x"].max()
y_min, y_max = plot_data["y"].min(), plot_data["y"].max()
z_min, z_max = plot_data["z"].min(), plot_data["z"].max()

x_padding = 0.05 * (x_max - x_min)
y_padding = 0.05 * (y_max - y_min)
z_padding = 0.05 * (z_max - z_min)

x_range = np.linspace(x_min - x_padding, x_max + x_padding, 30)
y_range = np.linspace(y_min - y_padding, y_max + y_padding, 30)

X_plane, Y_plane = np.meshgrid(x_range, y_range)

# Plane equation:
# a*x + b*y + c*z + intercept = 0
# Therefore:
# z = -(a*x + b*y + intercept) / c

if abs(c) < 1e-10:
    raise ValueError(
        "The fitted plane is nearly vertical. "
        "Cannot express the plane reliably as z = f(x, y)."
    )

Z_plane = -(
    a * X_plane + b * Y_plane + intercept_plot
) / c

fig = go.Figure()

# Plot boundary points
fig.add_trace(
    go.Scatter3d(
        x=plot_data["x"],
        y=plot_data["y"],
        z=plot_data["z"],
        mode="markers",
        name="Boundary points",
        marker=dict(
            size=4,
            opacity=0.85
        ),
        text=plot_data["point_id"],
        hovertemplate=(
            "Point ID: %{text}<br>"
            "x: %{x:.3f}<br>"
            "y: %{y:.3f}<br>"
            "z: %{z:.3f}<extra></extra>"
        )
    )
)

# Plot fitted LSR plane
fig.add_trace(
    go.Surface(
        x=X_plane,
        y=Y_plane,
        z=Z_plane,
        name="LSR plane",
        opacity=0.45,
        showscale=False,
        hovertemplate=(
            "LSR plane<br>"
            "x: %{x:.3f}<br>"
            "y: %{y:.3f}<br>"
            "z: %{z:.3f}<extra></extra>"
        )
    )
)

# Optional: plot removed outliers, if any exist
if len(outlier_data) > 0:
    fig.add_trace(
        go.Scatter3d(
            x=outlier_data["x"],
            y=outlier_data["y"],
            z=outlier_data["z"],
            mode="markers",
            name="Removed outliers",
            marker=dict(
                size=5,
                symbol="x",
                opacity=0.95
            ),
            text=outlier_data["point_id"],
            hovertemplate=(
                "Removed outlier<br>"
                "Point ID: %{text}<br>"
                "x: %{x:.3f}<br>"
                "y: %{y:.3f}<br>"
                "z: %{z:.3f}<extra></extra>"
            )
        )
    )

fig.update_layout(
    title=plot_title,
    scene=dict(
        xaxis_title="x / Easting (m)",        #<-- change axis title
        yaxis_title="y / Northing (m)",       #<-- change axis title
        zaxis_title="z / Elevation (m)",      #<-- change axis title
        xaxis=dict(range=[x_min - x_padding, x_max + x_padding]),
        yaxis=dict(range=[y_min - y_padding, y_max + y_padding]),
        zaxis=dict(range=[z_min - z_padding, z_max + z_padding]),
        aspectmode="data"
    ),
    legend=dict(
        x=0.02,
        y=0.98
    ),
    width=1000,
    height=800
)

interactive_plot_path = os.path.join(
    output_dir,
    "LSR_geological_boundary_interactive_3D.html"
)

fig.write_html(interactive_plot_path)
fig.show()
print(f"Interactive 3D Plotly figure: {interactive_plot_path}")


# ========================
# 10. Print Saved Files
# ========================
print("\nSaved outputs")
print("-------------")
print(f"Initial point residuals: {initial_points_path}")
print(f"Cleaned point residuals: {cleaned_points_path}")
print(f"Removed outliers: {outlier_points_path}")
print(f"Plane summary: {summary_path}")
print(f"Residual histogram: {residual_histogram_path}")


# Runtime Summary
end_time = time.perf_counter()
elapsed_time = end_time - start_time
elapsed_minutes = int(elapsed_time // 60)
elapsed_seconds = elapsed_time % 60

print("\nRuntime Summary")
print("---------------")
print(f"Total runtime: {elapsed_minutes} min {elapsed_seconds:.2f} sec")
----
