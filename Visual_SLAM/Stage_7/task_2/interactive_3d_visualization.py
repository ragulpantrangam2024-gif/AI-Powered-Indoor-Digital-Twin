import os
import numpy as np
import plotly.graph_objects as go


# ============================================================
# STAGE 7 - TASK 2
# INTERACTIVE 3D DIGITAL TWIN VISUALIZATION
# ============================================================

print("=" * 70)
print("STAGE 7 - TASK 2")
print("INTERACTIVE 3D DIGITAL TWIN VISUALIZATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STAGE_7_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_DIR = os.path.dirname(
    STAGE_7_DIR
)

STAGE_6_RESULTS = os.path.join(
    PROJECT_DIR,
    "Stage_6",
    "results"
)

RESULTS_DIR = os.path.join(
    STAGE_7_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


print("\nProject directory:")
print(PROJECT_DIR)

print("\nStage 6 results:")
print(STAGE_6_RESULTS)

print("\nStage 7 results:")
print(RESULTS_DIR)


# ============================================================
# INPUT FILES
# ============================================================

MAP_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task5_optimized_map_points.npy"
)

ROTATION_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task5_optimized_rotations.npy"
)

TRANSLATION_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task5_optimized_translations.npy"
)

TRAJECTORY_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task6_final_camera_trajectory.npy"
)


# ============================================================
# CHECK FILES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CHECKING INPUT FILES"
)

print(
    "=" * 70
)


required_files = [
    MAP_FILE,
    ROTATION_FILE,
    TRANSLATION_FILE,
    TRAJECTORY_FILE
]


for file_path in required_files:

    if not os.path.isfile(file_path):

        print(
            "\nERROR: File not found:"
        )

        print(
            file_path
        )

        raise SystemExit

    print(
        "[OK]",
        os.path.basename(file_path)
    )


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOADING STAGE 6 RECONSTRUCTION"
)

print(
    "=" * 70
)


map_points = np.load(
    MAP_FILE
)

rotations = np.load(
    ROTATION_FILE
)

translations = np.load(
    TRANSLATION_FILE
)

trajectory = np.load(
    TRAJECTORY_FILE
)


print(
    "\nMap points:",
    map_points.shape
)

print(
    "Rotations:",
    rotations.shape
)

print(
    "Translations:",
    translations.shape
)

print(
    "Trajectory:",
    trajectory.shape
)


# ============================================================
# VALIDATION
# ============================================================

if map_points.ndim != 2:

    raise ValueError(
        "Map points must be a 2D array."
    )


if map_points.shape[1] != 3:

    raise ValueError(
        "Map points must contain XYZ coordinates."
    )


if trajectory.ndim != 2:

    raise ValueError(
        "Trajectory must be a 2D array."
    )


if trajectory.shape[1] != 3:

    raise ValueError(
        "Trajectory must contain XYZ coordinates."
    )


# ============================================================
# REMOVE INVALID POINTS
# ============================================================

valid_points = np.all(
    np.isfinite(map_points),
    axis=1
)


map_points = map_points[
    valid_points
]


print(
    "\nValid map points:",
    len(map_points)
)


# ============================================================
# PRINT BASIC INFORMATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "RECONSTRUCTION INFORMATION"
)

print(
    "=" * 70
)


print(
    "\n3D Map"
)

print(
    "Points:",
    len(map_points)
)

print(
    "X range:",
    map_points[:, 0].min(),
    "to",
    map_points[:, 0].max()
)

print(
    "Y range:",
    map_points[:, 1].min(),
    "to",
    map_points[:, 1].max()
)

print(
    "Z range:",
    map_points[:, 2].min(),
    "to",
    map_points[:, 2].max()
)


print(
    "\nCamera trajectory"
)

print(
    "Positions:",
    len(trajectory)
)

print(
    "Start:",
    trajectory[0]
)

print(
    "End:",
    trajectory[-1]
)


# ============================================================
# CAMERA POSITION VALIDATION
# ============================================================

camera_positions = []


for i in range(
    len(rotations)
):

    R = rotations[i]

    t = translations[i]

    position = -R.T @ t

    camera_positions.append(
        position
    )


camera_positions = np.asarray(
    camera_positions
)


print(
    "\nCalculated camera positions:",
    camera_positions.shape
)


# ============================================================
# USE STORED TRAJECTORY
# ============================================================

# The Stage 6 final trajectory is the
# optimized camera trajectory.

camera_trajectory = trajectory.copy()


# ============================================================
# CREATE 3D MAP TRACE
# ============================================================

print(
    "\nCreating 3D map..."
)


map_trace = go.Scatter3d(

    x=map_points[:, 0],

    y=map_points[:, 1],

    z=map_points[:, 2],

    mode="markers",

    name="3D Map Points",

    marker=dict(

        size=2,

        opacity=0.55

    ),

    hovertemplate=(

        "X: %{x:.2f}<br>"

        "Y: %{y:.2f}<br>"

        "Z: %{z:.2f}"

        "<extra>3D Map Point</extra>"

    )
)


# ============================================================
# CAMERA TRAJECTORY TRACE
# ============================================================

print(
    "Creating camera trajectory..."
)


trajectory_trace = go.Scatter3d(

    x=camera_trajectory[:, 0],

    y=camera_trajectory[:, 1],

    z=camera_trajectory[:, 2],

    mode="lines+markers",

    name="Camera Trajectory",

    line=dict(

        width=5

    ),

    marker=dict(

        size=4

    ),

    hovertemplate=(

        "X: %{x:.2f}<br>"

        "Y: %{y:.2f}<br>"

        "Z: %{z:.2f}"

        "<extra>Camera Position</extra>"

    )
)


# ============================================================
# START POSITION
# ============================================================

start_position = camera_trajectory[0]


start_trace = go.Scatter3d(

    x=[start_position[0]],

    y=[start_position[1]],

    z=[start_position[2]],

    mode="markers+text",

    name="Start",

    marker=dict(

        size=10

    ),

    text=["START"],

    textposition="top center"

)


# ============================================================
# END POSITION
# ============================================================

end_position = camera_trajectory[-1]


end_trace = go.Scatter3d(

    x=[end_position[0]],

    y=[end_position[1]],

    z=[end_position[2]],

    mode="markers+text",

    name="End",

    marker=dict(

        size=10

    ),

    text=["END"],

    textposition="top center"

)


# ============================================================
# CAMERA POSITION MARKERS
# ============================================================

camera_marker_trace = go.Scatter3d(

    x=camera_trajectory[:, 0],

    y=camera_trajectory[:, 1],

    z=camera_trajectory[:, 2],

    mode="markers",

    name="Keyframes",

    marker=dict(

        size=5,

        symbol="diamond"

    ),

    hovertemplate=(

        "Keyframe: %{text}<br>"

        "X: %{x:.2f}<br>"

        "Y: %{y:.2f}<br>"

        "Z: %{z:.2f}"

        "<extra></extra>"

    ),

    text=[
        str(i)
        for i in range(
            len(camera_trajectory)
        )
    ]

)


# ============================================================
# CREATE FIGURE
# ============================================================

print(
    "Building interactive figure..."
)


fig = go.Figure(
    data=[
        map_trace,
        trajectory_trace,
        camera_marker_trace,
        start_trace,
        end_trace
    ]
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=dict(

        text=(
            "AI-Powered Indoor Digital Twin<br>"
            "<sup>"
            "Stage 7 - Stage 6 SLAM Reconstruction"
            "</sup>"
        ),

        x=0.5

    ),

    scene=dict(

        xaxis=dict(

            title="X",

            backgroundcolor="rgb(245,245,245)",

            showgrid=True

        ),

        yaxis=dict(

            title="Y",

            backgroundcolor="rgb(245,245,245)",

            showgrid=True

        ),

        zaxis=dict(

            title="Z",

            backgroundcolor="rgb(245,245,245)",

            showgrid=True

        ),

        aspectmode="data"

    ),

    legend=dict(

        x=0.01,

        y=0.99

    ),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=80

    )

)


# ============================================================
# ADD CAMERA TRAJECTORY INFORMATION
# ============================================================

displacement_vector = (
    camera_trajectory[-1]
    -
    camera_trajectory[0]
)


displacement = np.linalg.norm(
    displacement_vector
)


if len(camera_trajectory) > 1:

    motion = np.linalg.norm(
        np.diff(
            camera_trajectory,
            axis=0
        ),
        axis=1
    )

    path_length = np.sum(
        motion
    )

else:

    path_length = 0.0


if path_length > 0:

    directness = (
        displacement
        /
        path_length
    )

else:

    directness = 0.0


# ============================================================
# SAVE HTML
# ============================================================

html_file = os.path.join(
    RESULTS_DIR,
    "task2_interactive_digital_twin.html"
)


print(
    "\nSaving interactive visualization..."
)


fig.write_html(
    html_file,

    include_plotlyjs=True,

    full_html=True
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    RESULTS_DIR,
    "task2_visualization_statistics.txt"
)


with open(
    summary_file,
    "w"
) as f:

    f.write(
        "STAGE 7 - TASK 2\n"
    )

    f.write(
        "INTERACTIVE 3D DIGITAL TWIN VISUALIZATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "3D MAP\n"
    )

    f.write(
        f"Number of Points : "
        f"{len(map_points)}\n"
    )

    f.write(
        f"Map Shape : "
        f"{map_points.shape}\n"
    )

    f.write(
        f"X Range : "
        f"{map_points[:, 0].min():.6f} "
        f"to "
        f"{map_points[:, 0].max():.6f}\n"
    )

    f.write(
        f"Y Range : "
        f"{map_points[:, 1].min():.6f} "
        f"to "
        f"{map_points[:, 1].max():.6f}\n"
    )

    f.write(
        f"Z Range : "
        f"{map_points[:, 2].min():.6f} "
        f"to "
        f"{map_points[:, 2].max():.6f}\n"
    )

    f.write(
        "\nCAMERA TRAJECTORY\n"
    )

    f.write(
        f"Keyframes : "
        f"{len(camera_trajectory)}\n"
    )

    f.write(
        f"Start Position : "
        f"{camera_trajectory[0]}\n"
    )

    f.write(
        f"End Position : "
        f"{camera_trajectory[-1]}\n"
    )

    f.write(
        f"Displacement : "
        f"{displacement:.6f}\n"
    )

    f.write(
        f"Path Length : "
        f"{path_length:.6f}\n"
    )

    f.write(
        f"Directness : "
        f"{directness:.6f}\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TASK 2 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "\n3D Map Points:",
    len(map_points)
)

print(
    "Camera Keyframes:",
    len(camera_trajectory)
)

print(
    "Trajectory Displacement:",
    f"{displacement:.6f}"
)

print(
    "Trajectory Path Length:",
    f"{path_length:.6f}"
)

print(
    "Trajectory Directness:",
    f"{directness:.6f}"
)

print(
    "\nInteractive visualization saved to:"
)

print(
    html_file
)

print(
    "\nStatistics saved to:"
)

print(
    summary_file
)

print(
    "\nOpen the HTML file in Google Chrome"
)

print(
    "to interact with the 3D reconstruction."
)

print(
    "\nNo matplotlib windows were opened."
)