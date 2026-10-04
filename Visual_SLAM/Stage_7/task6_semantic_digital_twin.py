import os
import json
import numpy as np
import plotly.graph_objects as go


# ============================================================
# STAGE 7 - TASK 6
# FINAL SEMANTIC 3D DIGITAL TWIN
# ============================================================

print("=" * 70)
print("STAGE 7 - TASK 6")
print("FINAL SEMANTIC 3D DIGITAL TWIN")
print("=" * 70)


# ============================================================
# PROJECT DIRECTORIES
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

STAGE_7_RESULTS = os.path.join(
    STAGE_7_DIR,
    "results"
)


# ============================================================
# INPUT FILES
# ============================================================

# Stage 6 reconstruction
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


# Stage 7 semantic objects
SEMANTIC_FILE = os.path.join(
    STAGE_7_RESULTS,
    "task5_fused_semantic_objects.json"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_HTML = os.path.join(
    STAGE_7_RESULTS,
    "task6_final_semantic_digital_twin.html"
)

OUTPUT_STATS = os.path.join(
    STAGE_7_RESULTS,
    "task6_semantic_digital_twin_statistics.txt"
)


# ============================================================
# PRINT DIRECTORIES
# ============================================================

print("\nProject directory:")
print(PROJECT_DIR)

print("\nStage 6 results:")
print(STAGE_6_RESULTS)

print("\nStage 7 results:")
print(STAGE_7_RESULTS)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)


required_files = {

    "Stage 6 optimized map":
        MAP_FILE,

    "Stage 6 optimized rotations":
        ROTATION_FILE,

    "Stage 6 optimized translations":
        TRANSLATION_FILE,

    "Stage 6 final trajectory":
        TRAJECTORY_FILE,

    "Stage 7 fused semantic objects":
        SEMANTIC_FILE

}


all_found = True


for name, path in required_files.items():

    if os.path.exists(path):

        print(
            f"[OK] {name}"
        )

    else:

        print(
            f"[ERROR] Missing: {name}"
        )

        print(
            path
        )

        all_found = False


if not all_found:

    print("\nERROR: Required files are missing.")

    print(
        "\nPlease check the Stage 6 results folder."
    )

    raise SystemExit


# ============================================================
# LOAD OPTIMIZED MAP
# ============================================================

print("\n" + "=" * 70)
print("LOADING STAGE 6 OPTIMIZED MAP")
print("=" * 70)


map_points = np.load(
    MAP_FILE
)

map_points = np.asarray(
    map_points,
    dtype=np.float64
)


print(
    "Optimized 3D map points:",
    len(map_points)
)

print(
    "Map shape:",
    map_points.shape
)


# ============================================================
# LOAD ROTATIONS
# ============================================================

rotations = np.load(
    ROTATION_FILE
)

rotations = np.asarray(
    rotations,
    dtype=np.float64
)


print(
    "Rotation shape:",
    rotations.shape
)


# ============================================================
# LOAD TRANSLATIONS
# ============================================================

translations = np.load(
    TRANSLATION_FILE
)

translations = np.asarray(
    translations,
    dtype=np.float64
)


print(
    "Translation shape:",
    translations.shape
)


# ============================================================
# CALCULATE CAMERA POSITIONS
# ============================================================

print("\nCalculating camera positions...")


camera_positions = []


for R, t in zip(
    rotations,
    translations
):

    R = np.asarray(
        R,
        dtype=np.float64
    )

    t = np.asarray(
        t,
        dtype=np.float64
    ).reshape(3)


    # Camera center
    C = -R.T @ t

    camera_positions.append(
        C
    )


camera_positions = np.asarray(
    camera_positions
)


print(
    "Calculated camera positions:",
    camera_positions.shape
)


# ============================================================
# LOAD FINAL TRAJECTORY
# ============================================================

trajectory = np.load(
    TRAJECTORY_FILE
)

trajectory = np.asarray(
    trajectory,
    dtype=np.float64
)


print(
    "Final trajectory:",
    trajectory.shape
)


# ============================================================
# LOAD SEMANTIC OBJECTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING FUSED SEMANTIC OBJECTS")
print("=" * 70)


with open(
    SEMANTIC_FILE,
    "r"
) as f:

    semantic_objects = json.load(
        f
    )


print(
    "Fused semantic objects:",
    len(semantic_objects)
)


for obj in semantic_objects:

    print(
        f"{obj['object_id']:12s} | "
        f"{obj['class_name']:12s} | "
        f"Observations: "
        f"{obj['observation_count']}"
    )


# ============================================================
# TRAJECTORY STATISTICS
# ============================================================

start_position = trajectory[0]

end_position = trajectory[-1]


displacement_vector = (
    end_position
    -
    start_position
)


displacement = float(
    np.linalg.norm(
        displacement_vector
    )
)


if len(trajectory) > 1:

    frame_motion = np.linalg.norm(
        np.diff(
            trajectory,
            axis=0
        ),
        axis=1
    )

    path_length = float(
        np.sum(frame_motion)
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
# BUILD FIGURE
# ============================================================

print("\n" + "=" * 70)
print("BUILDING INTERACTIVE DIGITAL TWIN")
print("=" * 70)


fig = go.Figure()


# ============================================================
# 3D MAP
# ============================================================

fig.add_trace(

    go.Scatter3d(

        x=map_points[:, 0],

        y=map_points[:, 1],

        z=map_points[:, 2],

        mode="markers",

        marker=dict(

            size=1.5,

            opacity=0.40

        ),

        name="Optimized 3D Map",

        hovertemplate=(

            "Map Point<br>"
            "X: %{x:.2f}<br>"
            "Y: %{y:.2f}<br>"
            "Z: %{z:.2f}"
            "<extra></extra>"

        )

    )

)


# ============================================================
# CAMERA TRAJECTORY
# ============================================================

fig.add_trace(

    go.Scatter3d(

        x=trajectory[:, 0],

        y=trajectory[:, 1],

        z=trajectory[:, 2],

        mode="lines+markers",

        line=dict(
            width=5
        ),

        marker=dict(
            size=3
        ),

        name="Camera Trajectory",

        hovertemplate=(

            "Camera Position<br>"
            "X: %{x:.2f}<br>"
            "Y: %{y:.2f}<br>"
            "Z: %{z:.2f}"
            "<extra></extra>"

        )

    )

)


# ============================================================
# CAMERA START
# ============================================================

fig.add_trace(

    go.Scatter3d(

        x=[trajectory[0, 0]],

        y=[trajectory[0, 1]],

        z=[trajectory[0, 2]],

        mode="markers+text",

        marker=dict(

            size=9,

            symbol="diamond"

        ),

        text=["START"],

        textposition="top center",

        name="Camera Start"

    )

)


# ============================================================
# CAMERA END
# ============================================================

fig.add_trace(

    go.Scatter3d(

        x=[trajectory[-1, 0]],

        y=[trajectory[-1, 1]],

        z=[trajectory[-1, 2]],

        mode="markers+text",

        marker=dict(

            size=9,

            symbol="diamond"

        ),

        text=["END"],

        textposition="top center",

        name="Camera End"

    )

)


# ============================================================
# SEMANTIC OBJECT MARKERS
# ============================================================

symbols = {

    "bed": "square",

    "chair": "circle",

    "keyboard": "diamond",

    "laptop": "cross",

    "backpack": "triangle",

    "book": "star",

    "bottle": "hexagon",

    "cup": "pentagon",

    "dining table": "square"

}


for obj in semantic_objects:

    class_name = obj[
        "class_name"
    ]


    center = obj[
        "3d_center_median"
    ]


    x = center["x"]

    y = center["y"]

    z = center["z"]


    symbol = symbols.get(
        class_name,
        "circle"
    )


    observations = obj[
        "observation_count"
    ]


    confidence = obj[
        "mean_confidence"
    ]


    object_id = obj[
        "object_id"
    ]


    label = (
        f"{class_name} "
        f"({object_id})"
    )


    hover_text = (

        f"<b>{class_name}</b><br>"
        f"Object ID: {object_id}<br>"
        f"Observations: {observations}<br>"
        f"Confidence: {confidence:.3f}<br>"
        f"X: {x:.2f}<br>"
        f"Y: {y:.2f}<br>"
        f"Z: {z:.2f}"

    )


    fig.add_trace(

        go.Scatter3d(

            x=[x],

            y=[y],

            z=[z],

            mode="markers+text",

            marker=dict(

                size=13,

                symbol=symbol

            ),

            text=[label],

            textposition="top center",

            name=label,

            hovertemplate=(

                hover_text
                +
                "<extra></extra>"

            )

        )

    )


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=(

        "Semantic Indoor Digital Twin"
        "<br>"
        "<sup>"
        "Optimized 3D Reconstruction + "
        "Camera Trajectory + "
        "Semantic Objects"
        "</sup>"

    ),

    scene=dict(

        xaxis_title="X",

        yaxis_title="Y",

        zaxis_title="Z",

        aspectmode="data"

    ),

    legend=dict(

        title="Digital Twin Layers"

    ),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=80

    )

)


# ============================================================
# SAVE HTML
# ============================================================

print(
    "\nSaving interactive visualization..."
)


fig.write_html(

    OUTPUT_HTML,

    include_plotlyjs=True

)


# ============================================================
# SAVE STATISTICS
# ============================================================

with open(
    OUTPUT_STATS,
    "w"
) as f:

    f.write(
        "STAGE 7 - TASK 6\n"
    )

    f.write(
        "FINAL SEMANTIC 3D DIGITAL TWIN\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Optimized 3D Map Points: "
        f"{len(map_points)}\n"
    )

    f.write(
        f"Camera Keyframes: "
        f"{len(camera_positions)}\n"
    )

    f.write(
        f"Trajectory Positions: "
        f"{len(trajectory)}\n"
    )

    f.write(
        f"Semantic Objects: "
        f"{len(semantic_objects)}\n"
    )

    f.write(
        "\nTRAJECTORY\n"
    )

    f.write(
        f"Displacement: "
        f"{displacement:.6f}\n"
    )

    f.write(
        f"Path Length: "
        f"{path_length:.6f}\n"
    )

    f.write(
        f"Directness: "
        f"{directness:.6f}\n"
    )

    f.write(
        "\nSEMANTIC OBJECTS\n"
    )


    for obj in semantic_objects:

        center = obj[
            "3d_center_median"
        ]


        f.write(
            "\n"
        )

        f.write(
            f"Object ID: "
            f"{obj['object_id']}\n"
        )

        f.write(
            f"Class: "
            f"{obj['class_name']}\n"
        )

        f.write(
            f"Observations: "
            f"{obj['observation_count']}\n"
        )

        f.write(
            f"Mean Confidence: "
            f"{obj['mean_confidence']:.4f}\n"
        )

        f.write(

            "3D Center: "
            f"("
            f"{center['x']:.4f}, "
            f"{center['y']:.4f}, "
            f"{center['z']:.4f}"
            f")\n"

        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 6 COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nOptimized 3D Map Points:",
    len(map_points)
)


print(
    "Camera Keyframes:",
    len(camera_positions)
)


print(
    "Semantic Objects:",
    len(semantic_objects)
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
    "\nInteractive Digital Twin saved to:"
)

print(
    OUTPUT_HTML
)


print(
    "\nStatistics saved to:"
)

print(
    OUTPUT_STATS
)


print(
    "\nOpen the HTML file in Google Chrome."
)


print(
    "\nNo matplotlib windows were opened."
)


print("=" * 70)