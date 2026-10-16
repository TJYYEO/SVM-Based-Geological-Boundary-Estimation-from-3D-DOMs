+*In[ ]:*+
[source, ipython3]
----
# =============================================================================
# SVM geological-boundary estimation workflow
# =============================================================================
# Author: Thomas Yeo
# Updated: 16 October 2026
# Version: 1.1
#
# Description:
# This script was developed for mapping geological boundary using a linear
# SVM based on 3D spatial coordinate inputs.
# It includes model training, evaluation, structural orientation calculation,
# visualisation, and export of model results.
# =============================================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns
import plotly.graph_objs as go

# Set global plotting style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,
    'font.weight': 'bold'
})

# ========================
# 1. User Settings
# ========================
# User-defined labels (can be strings or numbers)
label_class1 = "ClassA" # <-- change this to your label
label_class2 = "ClassB" # <-- change this to your label

# Import .txt file containing space separated geographical cartesian coordinates (X Y Z).
file1 = '/Users/YourUserName/FileLocation/FileName.txt' # <-- Corresponds to ClassA
file2 = '/Users/YourUserName/FileLocation/FileName.txt' # <-- Corresponds to ClassB

# User-defined colours
colour_class1 = "green"      # ClassA
colour_class2 = "blue"       # ClassB
colour_support = "red"       # Support vectors

# Output folder
output_dir = "/Users/YourUserName/FileLocation" #<-- change this to your save folder
os.makedirs(output_dir, exist_ok=True) #<-- True or False

# SVM parameter
C_value = 10**2              # Default is set to C=100
random_state = 42
test_size = 0.30

# Start Timer
start_time = time.perf_counter()

# ========================
# 2. Load and Prepare Data
# ========================
# sep=r"\s+" allows either spaces or tabs
data1 = pd.read_csv(file1, sep=r"\s+", header=None, names=["x", "y", "z"])
data2 = pd.read_csv(file2, sep=r"\s+", header=None, names=["x", "y", "z"])

data1['label'] = label_class1
data2['label'] = label_class2

data = pd.concat([data1, data2], ignore_index=True)
data["point_id"] = np.arange(len(data))

X = data[["x", "y", "z"]]
y = data["label"]

# ========================
# 3. Standardise Features
# ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 4. Train-Test Split
# ========================
indices = data.index.values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled,
    y,
    indices,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# ========================
# 5. Train SVM Classifier (Soft Margin)
# ========================
svm = SVC(
    kernel="linear",
    C=C_value,
    random_state=random_state
)

svm.fit(X_train, y_train)

# ========================
# 6. Predict and Evaluate
# ========================
y_pred_test = svm.predict(X_test)
y_pred_all = svm.predict(X_scaled)

metrics = {
    "Accuracy": accuracy_score(
        y_test,
        y_pred_test
    ),
    "Precision": precision_score(
        y_test,
        y_pred_test,
        pos_label=label_class2,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        y_pred_test,
        pos_label=label_class2,
        zero_division=0
    ),
    "F1 Score": f1_score(
        y_test,
        y_pred_test,
        pos_label=label_class2,
        zero_division=0
    )
}

cm = confusion_matrix(
    y_test,
    y_pred_test,
    labels=[label_class1, label_class2]
)

cross_val_scores = cross_val_score(
    svm,
    X_scaled,
    y,
    cv=5
)

# ========================
# 7. Extract Decision Plane Parameters
# ========================
coef_scaled = svm.coef_[0]
intercept_scaled = svm.intercept_[0]

# Convert plane coefficients back to original coordinate system
coef_original = coef_scaled / scaler.scale_
intercept_original = intercept_scaled - np.dot(
    coef_scaled,
    scaler.mean_ / scaler.scale_
)

a, b, c = coef_original
normal_vector = np.array([a, b, c])

norm_w = np.linalg.norm(normal_vector)
margin = 1.0 / norm_w

# Signed perpendicular distance from each point to the decision plane
signed_distances = (X.values @ coef_original + intercept_original) / norm_w

# ========================
# 8. Compute Dip and Dip Direction
# Geological convention:
# Dip direction azimuth: 0° = North, clockwise positive
# Coordinates assumed: x = Easting, y = Northing, z = Elevation
# ========================
horizontal_length = np.sqrt(a**2 + b**2)
dip = np.degrees(np.arctan2(horizontal_length, abs(c)))

z_axis = np.array([0.0, 0.0, 1.0])
strike = np.cross(normal_vector, z_axis)

if np.linalg.norm(strike) == 0:
    raise ValueError("Plane is horizontal; dip direction is undefined.")

dip_vec = np.cross(normal_vector, strike)
dip_vec = dip_vec / np.linalg.norm(dip_vec)

# Ensure down-dip vector points downward

if dip_vec[2] > 0:
    dip_vec = -dip_vec

dip_direction = (
    np.degrees(np.arctan2(dip_vec[0], dip_vec[1])) + 360
) % 360

# ========================
# 9. Export SVM-Classified Points
# ========================
classified_points = data.copy()
classified_points["predicted_label"] = y_pred_all
classified_points["signed_distance_to_plane"] = signed_distances
classified_points["absolute_distance_to_plane"] = np.abs(signed_distances)

classified_points_path = os.path.join(
    output_dir,
    "SVM_classified_points.csv"
)

classified_points.to_csv(classified_points_path, index=False)

# ========================
# 10. Export Support Vectors
# ========================
support_vectors_original = scaler.inverse_transform(svm.support_vectors_)

# svm.support_ gives the indices of support vectors within X_train
support_train_indices = svm.support_
support_original_indices = idx_train[support_train_indices]

support_vectors_df = pd.DataFrame(
    support_vectors_original,
    columns=["x", "y", "z"]
)

support_vectors_df["original_point_id"] = data.loc[
    support_original_indices,
    "point_id"
].values

support_vectors_df["original_label"] = data.loc[
    support_original_indices,
    "label"
].values

support_vectors_df["signed_distance_to_plane"] = (
    support_vectors_df[["x", "y", "z"]].values @ coef_original
    + intercept_original
) / norm_w

support_vectors_path = os.path.join(
    output_dir,
    "SVM_support_vectors.csv"
)

support_vectors_df.to_csv(support_vectors_path, index=False)

support_vector_summary = pd.DataFrame({
    "class": svm.classes_,
    "number_of_support_vectors": svm.n_support_
})

support_vector_summary_path = os.path.join(
    output_dir,
    "SVM_support_vector_summary.csv"
)

support_vector_summary.to_csv(
    support_vector_summary_path,
    index=False
)

# ========================
# 11. Export Metrics and Plane Summary
# ========================
metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Cross-validation mean accuracy",
        "Cross-validation standard deviation",
        "Margin",
        "Mean signed distance",
        "Minimum signed distance",
        "Maximum signed distance",
        "Standard deviation of signed distance",
        "Dip",
        "Dip direction",
        "Number of support vectors"
    ],

    "Value": [
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1 Score"],
        cross_val_scores.mean(),
        cross_val_scores.std(),
        margin,
        signed_distances.mean(),
        signed_distances.min(),
        signed_distances.max(),
        signed_distances.std(),
        dip,
        dip_direction,
        len(support_vectors_df)
    ]
})

metrics_path = os.path.join(
    output_dir,
    "SVM_metrics_summary.csv"
)

metrics_df.to_csv(metrics_path, index=False)

plane_df = pd.DataFrame({
    "Parameter": [
        "intercept",
        "normal_x",
        "normal_y",
        "normal_z",
        "dip",
        "dip_direction",
        "down_dip_vector_x",
        "down_dip_vector_y",
        "down_dip_vector_z",
        "margin"
    ],
    "Value": [
        intercept_original,
        normal_vector[0],
        normal_vector[1],
        normal_vector[2],
        dip,
        dip_direction,
        dip_vec[0],
        dip_vec[1],
        dip_vec[2],
        margin
    ]
})

plane_path = os.path.join(
    output_dir,
    "SVM_plane_parameters.csv"
)

plane_df.to_csv(plane_path, index=False)

cm_df = pd.DataFrame(
    cm,
    index=[f"True_{label_class1}", f"True_{label_class2}"],
    columns=[f"Predicted_{label_class1}", f"Predicted_{label_class2}"]
)

cm_path = os.path.join(
    output_dir,
    "SVM_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

# ========================
# 12. Create Decision and Margin Planes
# ========================
xx, yy = np.meshgrid(
    np.linspace(X["x"].min(), X["x"].max(), 30),
    np.linspace(X["y"].min(), X["y"].max(), 30)
)

if abs(c) < 1e-12:
    raise ValueError(
        "The plane coefficient c is close to zero. "
        "The plane cannot be plotted as z = f(x, y)."
    )

zz = (-a * xx - b * yy - intercept_original) / c

# Margin planes in original coordinates.
# Since the geometric margin is defined by w.x + b = ±1,
# the ±1 offsets are applied to the original unnormalised plane equation.
zz_plus = (-a * xx - b * yy - (intercept_original - 1)) / c
zz_minus = (-a * xx - b * yy - (intercept_original + 1)) / c

# ========================
# 13. Matplotlib 3D Plot
# ========================
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X[y == label_class1]["x"],
    X[y == label_class1]["y"],
    X[y == label_class1]["z"],
    color=colour_class1,
    label=str(label_class1),
    s=6,
    alpha=0.85
)

ax.scatter(
    X[y == label_class2]["x"],
    X[y == label_class2]["y"],
    X[y == label_class2]["z"],
    color=colour_class2,
    label=str(label_class2),
    s=6,
    alpha=0.85
)

ax.scatter(
    support_vectors_original[:, 0],
    support_vectors_original[:, 1],
    support_vectors_original[:, 2],
    color=colour_support,
    s=25,
    edgecolors=colour_support,
    label="Support vectors"
)

ax.plot_surface(xx, yy, zz, color="gray", alpha=0.7, label='Decision Plane')
#ax.plot_surface(xx, yy, zz_plus, color="cyan", alpha=0.3)      #<-- positive margin
#ax.plot_surface(xx, yy, zz_minus, color="magenta", alpha=0.3)  #<-- negative margin

# Axis labels
ax.set_xlabel("Easting (m)", fontsize=14, fontweight="bold", labelpad=8)     #<-- change axis title
ax.set_ylabel("Northing (m)", fontsize=14, fontweight="bold", labelpad=8)    #<-- change axis title
ax.set_zlabel("Elevation (m)", fontsize=14, fontweight="bold", labelpad=-1)  #<-- change axis title

ax.tick_params(axis="x", labelsize=12, pad=1)
ax.tick_params(axis="y", labelsize=12, pad=1)
ax.tick_params(axis="z", labelsize=12, pad=1)

# Manually set axis ranges; adjust if needed
#ax.set_xlim(22910, 23110)         #<-- change this to your Easting range
#ax.set_ylim(-172705, -172645)     #<-- change this to your Northing range
#ax.set_zlim(600, 730)             #<-- change this to your Elevation range

# Viewing angle; adjust if needed 
#ax.view_init(elev=25, azim=-45)   #<-- change this to your viewing angle

ax.set_title("Geological boundary", fontsize=20, fontweight="bold", pad=5) #<-- change this to your title

# Legend
ax.legend(
    fontsize=12,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98)
)

fig_path_png = os.path.join(output_dir, "Geological boundary.png")

plt.savefig(fig_path_png, dpi=600)
plt.show()

# ========================
# 14. Confusion Matrix Figure
# ========================
plt.figure(figsize=(8, 6))

# Percentages are calculated relative to all observations in the test set.
cm_percent = cm / cm.sum() * 100
cm_annotations = np.array([
    [f"{count}\n({percentage:.1f}%)" for count, percentage in zip(count_row, percentage_row)]
    for count_row, percentage_row in zip(cm, cm_percent)
])

sns.heatmap(
    cm,
    annot=cm_annotations,
    fmt="",
    cmap="Blues",
    xticklabels=[label_class1, label_class2],
    yticklabels=[label_class1, label_class2]
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")

cm_fig_png = os.path.join(output_dir, "SVM_confusion_matrix.png")

plt.savefig(cm_fig_png, dpi=600, bbox_inches="tight")
plt.show()

# ========================
# 15. Cross-Validation Plot
# ========================
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, 6),
    cross_val_scores,
    marker="o",
    label="CV score"
)

plt.axhline(
    y=cross_val_scores.mean(),
    color="r",
    linestyle="--",
    label=f"Mean: {cross_val_scores.mean():.4f}"
)

plt.xlabel("Fold number")
plt.ylabel("Accuracy")
plt.title("Five-fold Cross-validation Scores")
plt.legend()
plt.grid(True)

cv_fig_png = os.path.join(output_dir, "SVM_cross_validation_scores.png")

plt.savefig(cv_fig_png, dpi=600, bbox_inches="tight")
plt.show()

# ========================
# 16. Interactive Plotly 3D Figure
# ========================
scatter0 = go.Scatter3d(
    x=X[y == label_class1]["x"],
    y=X[y == label_class1]["y"],
    z=X[y == label_class1]["z"],
    mode="markers",
    marker=dict(size=5, color=colour_class1),
    name=str(label_class1)
)

scatter1 = go.Scatter3d(
    x=X[y == label_class2]["x"],
    y=X[y == label_class2]["y"],
    z=X[y == label_class2]["z"],
    mode="markers",
    marker=dict(size=5, color=colour_class2),
    name=str(label_class2)
)


support = go.Scatter3d(
    x=support_vectors_original[:, 0],
    y=support_vectors_original[:, 1],
    z=support_vectors_original[:, 2],
    mode="markers",
    marker=dict(size=5, color=colour_support),
    name="Support vectors"
)

surface_decision = go.Surface(
    x=xx,
    y=yy,
    z=zz,
    opacity=0.7,
    colorscale="Greys",
    showscale=False,
    name="Decision plane"
)

surface_margin_plus = go.Surface(
    x=xx,
    y=yy,
    z=zz_plus,
    opacity=0.3,
    colorscale=[[0, "cyan"], [1, "cyan"]],
    showscale=False,
    name="+Margin"
)

surface_margin_minus = go.Surface(
    x=xx,
    y=yy,
    z=zz_minus,
    opacity=0.3,
    colorscale=[[0, "magenta"], [1, "magenta"]],
    showscale=False,
    name="-Margin"
)

fig_plotly = go.Figure(
    data=[
        scatter0,
        scatter1,
        support,
        surface_decision,
        surface_margin_plus,
        surface_margin_minus
    ]
)

fig_plotly.update_layout(
    scene=dict(
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        zaxis_title="Elevation (m)",

        # This removes vertical exaggeration.
        # Axis lengths are proportional to the real data ranges.
        # aspectmode="data"
    ),
    title="Geological boundary with margins" #<-- change this to your title 
)

plotly_html_path = os.path.join(
    output_dir,
    "SVM_geological_boundary_interactive.html"
)

fig_plotly.write_html(plotly_html_path)
fig_plotly.show()

# ========================
# 17. Print Summary
# ========================
print("\nModel Evaluation Metrics")
print("------------------------")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print(f"Cross-validation mean accuracy: {cross_val_scores.mean():.4f}")
print(f"Cross-validation standard deviation: {cross_val_scores.std():.4f}")

print("\nFault Plane Orientation")
print("-----------------------")
print(f"Dip: {dip:.2f}°")
print(f"Dip direction: {dip_direction:.2f}°")
print(f"Normal vector: ({a:.7f}, {b:.7f}, {c:.7f})")
print(f"Down-dip vector: ({dip_vec[0]:.7f}, {dip_vec[1]:.7f}, {dip_vec[2]:.7f})")

print("\nPlane Equation")
print("--------------")
print(
    f"{a:.7f} * x + {b:.7f} * y + {c:.7f} * z "
    f"+ {intercept_original:.7f} = 0"
)

print("\nDistance Summary")
print("----------------")
print(f"Margin: ± {margin:.4f}")
print(f"Mean signed distance: {signed_distances.mean():.4f}")
print(f"Minimum signed distance: {signed_distances.min():.4f}")
print(f"Maximum signed distance: {signed_distances.max():.4f}")
print(f"Standard deviation: {signed_distances.std():.4f}")

print("\nSupport Vectors")
print("---------------")
print(f"Total number of support vectors: {len(support_vectors_df)}")
print(support_vector_summary)

print("\nSaved outputs")
print("-------------")
print(f"Classified points: {classified_points_path}")
print(f"Support vectors: {support_vectors_path}")
print(f"Support vector summary: {support_vector_summary_path}")
print(f"Metrics summary: {metrics_path}")
print(f"Plane parameters: {plane_path}")
print(f"Confusion matrix CSV: {cm_path}")
print(f"3D figure PNG: {fig_path_png}")
print(f"Confusion matrix figure PNG: {cm_fig_png}")
print(f"Cross-validation figure PNG: {cv_fig_png}")
print(f"Interactive 3D figure HTML: {plotly_html_path}")

# ========================
# 18. Runtime Summary
# ========================

end_time = time.perf_counter()
elapsed_time = end_time - start_time

elapsed_minutes = int(elapsed_time // 60)
elapsed_seconds = elapsed_time % 60

print("\nRuntime Summary")
print("---------------")
print(f"Total runtime: {elapsed_minutes} min {elapsed_seconds:.2f} sec")
----
