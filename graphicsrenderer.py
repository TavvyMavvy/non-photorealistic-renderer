"""
graphicsrenderer.py
-------------------

Coursework renderer for drawing a 3D skull model with non-photorealistic rendering.

What it produces:
    1. A normal shaded reference image.
    2. A toon-shaded image.
    3. A contour-only line drawing.
    4. A toon + contour result.
    5. An optional manga-style creative result.

Install:
    pip install numpy pillow trimesh

Run on the coursework skull model:
    python3 graphicsrenderer.py --mesh "skull model/12140_Skull_v3_L2.obj" --out "outputs/skull_final_simple" --creative --eye "1.0,-3.0,0.3" --up "0,0,1"

This version is intentionally written in a clear step-by-step style. The program
follows the same order as the rendering pipeline:

    load mesh -> setup camera -> shade triangles -> find contours -> draw lines

The important coursework idea is the contour test. An edge becomes a contour when
its two neighbouring faces disagree about whether they face the camera. This is a
practical mesh version of the DeCarlo-style idea that contours occur where the
surface just grazes the line of sight.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import trimesh


# -----------------------------------------------------------------------------
# Small settings that are useful to change while experimenting
# -----------------------------------------------------------------------------

BACKGROUND_COLOUR = (245, 242, 232)
BONE_COLOUR = (214, 198, 168)
LINE_COLOUR = (0, 0, 0)
DEFAULT_LIGHT = np.array([0.2, -1.0, 0.6], dtype=float)


# -----------------------------------------------------------------------------
# Basic vector and mesh helpers
# -----------------------------------------------------------------------------


def normalise(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a vector, or array of vectors, scaled to length 1.

    The small eps value prevents division by zero if a vector has length 0.
    """
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(length, eps)


def calculate_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Calculate one normal vector for each triangular face.

    For a triangle with corners a, b and c, the cross product of two triangle
    edges gives a vector perpendicular to the triangle.
    """
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    return normalise(normals)


def centre_and_scale_mesh(vertices: np.ndarray) -> np.ndarray:
    """Move the model to the origin and scale it to roughly unit size.

    This makes the camera settings easier to reuse even if the OBJ file uses a
    very large or very small coordinate system.
    """
    centre = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = vertices - centre

    biggest_distance = np.max(np.linalg.norm(vertices, axis=1))
    if biggest_distance > 0:
        vertices = vertices / biggest_distance

    return vertices


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_mesh(path: str) -> Mesh:
    """Load an OBJ/PLY/STL mesh and do basic cleanup.

    I used trimesh for loading because it handles OBJ files and gives direct
    access to the vertices and triangle faces.
    """
    mesh = trimesh.load(path, force="mesh")

    # Some OBJ files load as a Scene if they contain multiple objects. The rest
    # of this renderer expects one mesh, so combine all pieces into one object.
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    # Basic cleanup. These checks are written defensively because Trimesh has
    # changed some method names between versions.
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()

    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    vertices = centre_and_scale_mesh(vertices)
    return Mesh(vertices, faces)


# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    width: int = 900
    height: int = 900
    fov_degrees: float = 40.0

    def get_basis_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the camera's right, up and forward directions."""
        forward = normalise(self.target - self.eye)
        right = normalise(np.cross(forward, self.up))
        true_up = normalise(np.cross(right, forward))
        return right, true_up, forward

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points into 2D screen coordinates.

        Returns:
            screen_points: x/y pixel positions.
            depth_values: distance in front of the camera.
        """
        right, up, forward = self.get_basis_vectors()

        # Put points into the camera's coordinate system.
        relative_points = points - self.eye[None, :]
        camera_x = relative_points @ right
        camera_y = relative_points @ up
        camera_z = relative_points @ forward

        # Perspective projection. Larger z means further from camera, so x and y
        # are divided by z.
        focal_length = 1.0 / math.tan(math.radians(self.fov_degrees) / 2.0)
        safe_z = np.maximum(camera_z, 1e-6)
        x_ndc = focal_length * camera_x / safe_z
        y_ndc = focal_length * camera_y / safe_z

        # Convert from normalised coordinates to image pixels.
        x_pixel = (x_ndc + 1.0) * 0.5 * (self.width - 1)
        y_pixel = (1.0 - (y_ndc + 1.0) * 0.5) * (self.height - 1)

        screen_points = np.column_stack([x_pixel, y_pixel])
        return screen_points, camera_z


# -----------------------------------------------------------------------------
# Triangle rasterisation and toon shading
# -----------------------------------------------------------------------------


def barycentric_weights(px: np.ndarray, py: np.ndarray, triangle: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find barycentric weights for points inside a 2D triangle.

    Barycentric weights tell us how much of each triangle vertex contributes to
    a pixel. Here I use them mainly for deciding if a pixel lies inside a triangle
    and for interpolating depth values.
    """
    x0, y0 = triangle[0]
    x1, y1 = triangle[1]
    x2, y2 = triangle[2]

    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-12:
        return (
            np.full(px.shape, np.nan),
            np.full(px.shape, np.nan),
            np.full(px.shape, np.nan),
        )

    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def toon_band(value: float, levels: int) -> float:
    """Turn smooth lighting into a small number of flat colour bands."""
    value = float(np.clip(value, 0.0, 1.0))
    if levels <= 1:
        return value
    band = min(math.floor(value * levels), levels - 1)
    return band / max(levels - 1, 1)


def render_mesh(
    mesh: Mesh,
    camera: Camera,
    mode: str,
    toon_levels: int = 4,
    base_colour: Tuple[int, int, int] = BONE_COLOUR,
    background_colour: Tuple[int, int, int] = BACKGROUND_COLOUR,
) -> Tuple[Image.Image, np.ndarray]:
    """Render the mesh using a simple software z-buffer.

    The renderer loops over triangles, fills the pixels inside each projected
    triangle, and uses a depth buffer so nearer triangles cover further ones.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    width = camera.width
    height = camera.height

    screen_vertices, vertex_depths = camera.project(vertices)
    normals = calculate_face_normals(vertices, faces)
    light_direction = normalise(DEFAULT_LIGHT)

    image_array = np.zeros((height, width, 3), dtype=float)
    image_array[:, :, :] = np.array(background_colour, dtype=float)
    depth_buffer = np.full((height, width), np.inf, dtype=float)

    for face_number, face in enumerate(faces):
        triangle = screen_vertices[face]
        triangle_depth = vertex_depths[face]

        # Skip triangles behind the camera.
        if np.any(triangle_depth <= 0):
            continue

        # Only check pixels in the triangle's bounding box.
        min_x = max(int(math.floor(np.min(triangle[:, 0]))), 0)
        max_x = min(int(math.ceil(np.max(triangle[:, 0]))), width - 1)
        min_y = max(int(math.floor(np.min(triangle[:, 1]))), 0)
        max_y = min(int(math.ceil(np.max(triangle[:, 1]))), height - 1)

        if min_x > max_x or min_y > max_y:
            continue

        x_values = np.arange(min_x, max_x + 1) + 0.5
        y_values = np.arange(min_y, max_y + 1) + 0.5
        px, py = np.meshgrid(x_values, y_values)

        w0, w1, w2 = barycentric_weights(px, py, triangle)
        inside_triangle = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)

        if not np.any(inside_triangle):
            continue

        pixel_depth = w0 * triangle_depth[0] + w1 * triangle_depth[1] + w2 * triangle_depth[2]

        local_y, local_x = np.nonzero(inside_triangle)
        image_y = local_y + min_y
        image_x = local_x + min_x
        candidate_depth = pixel_depth[inside_triangle]

        is_closer = candidate_depth < depth_buffer[image_y, image_x]
        if not np.any(is_closer):
            continue

        normal = normals[face_number]
        diffuse = max(float(normal @ light_direction), 0.0)
        ambient = 0.30

        if mode == "phong":
            shade = ambient + 0.82 * diffuse
        elif mode == "toon":
            shade = ambient + 0.82 * toon_band(diffuse, toon_levels)
        elif mode == "white":
            shade = 1.0
        else:
            raise ValueError(f"Unknown render mode: {mode}")

        colour = np.array(base_colour, dtype=float) * shade

        final_y = image_y[is_closer]
        final_x = image_x[is_closer]
        depth_buffer[final_y, final_x] = candidate_depth[is_closer]
        image_array[final_y, final_x, :] = np.clip(colour, 0, 255)

    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    return image, depth_buffer


# -----------------------------------------------------------------------------
# Contour extraction
# -----------------------------------------------------------------------------


def build_edge_table(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """Build a table saying which faces touch each mesh edge."""
    edge_table: Dict[Tuple[int, int], List[int]] = {}

    for face_index, face in enumerate(faces):
        edges = [
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ]

        for a, b in edges:
            edge = tuple(sorted((int(a), int(b))))
            if edge not in edge_table:
                edge_table[edge] = []
            edge_table[edge].append(face_index)

    return edge_table


def find_contour_edges(mesh: Mesh, camera: Camera) -> List[Tuple[int, int]]:
    """Find view-dependent contour edges.

    A mesh edge is drawn if one adjacent face is front-facing and the other is
    back-facing. This is the practical triangle-mesh version of a contour: the
    surface is turning away from the camera and grazing the line of sight.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    normals = calculate_face_normals(vertices, faces)
    face_centres = vertices[faces].mean(axis=1)

    view_directions = normalise(camera.eye[None, :] - face_centres)
    facing_values = np.sum(normals * view_directions, axis=1)

    edge_table = build_edge_table(faces)
    contour_edges: List[Tuple[int, int]] = []

    for (a, b), touching_faces in edge_table.items():
        # Boundary edges caused visible seam lines on the skull model, so I chose
        # to ignore them and only use edges between two triangles.
        if len(touching_faces) != 2:
            continue

        first_face, second_face = touching_faces
        first_value = facing_values[first_face]
        second_value = facing_values[second_face]

        if first_value == 0 or second_value == 0 or first_value * second_value < 0:
            contour_edges.append((a, b))

    return contour_edges


# -----------------------------------------------------------------------------
# Drawing projected lines with visibility testing
# -----------------------------------------------------------------------------


def visible_line_pieces(
    start_2d: np.ndarray,
    end_2d: np.ndarray,
    start_depth: float,
    end_depth: float,
    depth_buffer: np.ndarray,
    sample_count: int = 80,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Split a line into visible pieces using the depth buffer."""
    height, width = depth_buffer.shape
    depth_bias = 0.012

    t_values = np.linspace(0.0, 1.0, sample_count)
    x_values = (1.0 - t_values) * start_2d[0] + t_values * end_2d[0]
    y_values = (1.0 - t_values) * start_2d[1] + t_values * end_2d[1]
    z_values = (1.0 - t_values) * start_depth + t_values * end_depth

    visible = []
    for i in range(sample_count):
        x = int(round(x_values[i]))
        y = int(round(y_values[i]))

        if 0 <= x < width and 0 <= y < height and z_values[i] > 0:
            visible.append(z_values[i] <= depth_buffer[y, x] + depth_bias)
        else:
            visible.append(False)

    pieces: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    start_index = None

    for i, is_visible in enumerate(visible):
        if is_visible and start_index is None:
            start_index = i

        reached_end = i == sample_count - 1
        if start_index is not None and ((not is_visible) or reached_end):
            end_index = i if is_visible and reached_end else i - 1

            # Ignore tiny pieces because they look like noise.
            if end_index - start_index >= 2:
                pieces.append(
                    (
                        (float(x_values[start_index]), float(y_values[start_index])),
                        (float(x_values[end_index]), float(y_values[end_index])),
                    )
                )
            start_index = None

    return pieces


def draw_contours(
    image: Image.Image,
    mesh: Mesh,
    contour_edges: List[Tuple[int, int]],
    camera: Camera,
    depth_buffer: np.ndarray,
    colour: Tuple[int, int, int] = LINE_COLOUR,
    line_width: int = 2,
) -> None:
    """Project contour edges and draw only the visible parts."""
    draw = ImageDraw.Draw(image)

    for a, b in contour_edges:
        points = np.vstack([mesh.vertices[a], mesh.vertices[b]])
        screen_points, depths = camera.project(points)

        if np.any(depths <= 0):
            continue

        pieces = visible_line_pieces(
            screen_points[0],
            screen_points[1],
            float(depths[0]),
            float(depths[1]),
            depth_buffer,
        )

        for start, end in pieces:
            draw.line([start, end], fill=colour, width=line_width)


# -----------------------------------------------------------------------------
# Creative contribution: manga-style hatching
# -----------------------------------------------------------------------------


def draw_manga_hatching(
    image: Image.Image,
    mesh: Mesh,
    camera: Camera,
    depth_buffer: np.ndarray,
) -> None:
    """Add manga-style hatch strokes in shadowed areas.

    This was my creative extension. It uses the same diffuse lighting value as
    toon shading. Bright triangles are left alone. Darker triangles receive
    diagonal hatching, with cross-hatching in the darkest regions.
    """
    draw = ImageDraw.Draw(image)
    vertices = mesh.vertices
    faces = mesh.faces
    normals = calculate_face_normals(vertices, faces)
    screen_vertices, vertex_depths = camera.project(vertices)
    light_direction = normalise(DEFAULT_LIGHT)

    for face_number, face in enumerate(faces):
        normal = normals[face_number]
        diffuse = max(float(normal @ light_direction), 0.0)

        # Only draw hatching on shadowed triangles.
        if diffuse > 0.75:
            continue

        triangle = screen_vertices[face]
        triangle_depth = vertex_depths[face]

        if np.any(triangle_depth <= 0):
            continue

        min_x = max(int(math.floor(np.min(triangle[:, 0]))), 0)
        max_x = min(int(math.ceil(np.max(triangle[:, 0]))), camera.width - 1)
        min_y = max(int(math.floor(np.min(triangle[:, 1]))), 0)
        max_y = min(int(math.ceil(np.max(triangle[:, 1]))), camera.height - 1)

        if min_x > max_x or min_y > max_y:
            continue

        # Smaller spacing means denser hatching.
        if diffuse > 0.50:
            spacing = 14
        elif diffuse > 0.30:
            spacing = 10
        else:
            spacing = 7

        line_length = 10

        for y in range(min_y + spacing // 2, max_y, spacing):
            for x in range(min_x + spacing // 2, max_x, spacing):
                w0, w1, w2 = barycentric_weights(
                    np.array([[x + 0.5]], dtype=float),
                    np.array([[y + 0.5]], dtype=float),
                    triangle,
                )
                weights = np.array([w0[0, 0], w1[0, 0], w2[0, 0]])

                if np.any(weights < -1e-6):
                    continue

                z = float(weights @ triangle_depth)
                if z > depth_buffer[y, x] + 0.015:
                    continue

                # First diagonal stroke.
                draw.line(
                    [(x - line_length, y + line_length // 2), (x + line_length, y - line_length // 2)],
                    fill=LINE_COLOUR,
                    width=1,
                )

                # Cross-hatching in darker areas.
                if diffuse < 0.32:
                    draw.line(
                        [(x - line_length, y - line_length // 2), (x + line_length, y + line_length // 2)],
                        fill=LINE_COLOUR,
                        width=1,
                    )

                # Extra vertical mark for the darkest areas.
                if diffuse < 0.18:
                    draw.line(
                        [(x, y - line_length // 2), (x, y + line_length // 2)],
                        fill=LINE_COLOUR,
                        width=1,
                    )


# -----------------------------------------------------------------------------
# Main rendering workflow
# -----------------------------------------------------------------------------


def render_all_outputs(mesh: Mesh, camera: Camera, output_prefix: str, creative: bool, toon_levels: int) -> None:
    """Run the complete coursework pipeline and save all output images."""
    output_path = Path(output_prefix)
    output_folder = output_path.parent
    output_name = output_path.name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: normal smooth-looking reference render.
    reference_image, _ = render_mesh(mesh, camera, mode="phong", toon_levels=toon_levels)
    reference_image.save(output_folder / f"{output_name}_phong_reference.png")

    # Step 2: toon shading, which is the minimum coursework level.
    toon_image, toon_depth = render_mesh(mesh, camera, mode="toon", toon_levels=toon_levels)
    toon_image.save(output_folder / f"{output_name}_toon.png")

    # Step 3: find contour edges once, then reuse them for all contour images.
    contour_edges = find_contour_edges(mesh, camera)

    # Step 4: draw contours on a plain white model.
    white_image, white_depth = render_mesh(
        mesh,
        camera,
        mode="white",
        base_colour=(242, 240, 232),
        background_colour=(248, 246, 238),
    )
    draw_contours(white_image, mesh, contour_edges, camera, white_depth, line_width=2)
    white_image.save(output_folder / f"{output_name}_contours.png")

    # Step 5: draw the same contours over the toon shaded image.
    toon_with_contours = toon_image.copy()
    draw_contours(toon_with_contours, mesh, contour_edges, camera, toon_depth, line_width=2)
    toon_with_contours.save(output_folder / f"{output_name}_toon_contours.png")

    # Step 6: optional creative contribution.
    if creative:
        creative_image = toon_image.copy()
        draw_manga_hatching(creative_image, mesh, camera, toon_depth)
        draw_contours(creative_image, mesh, contour_edges, camera, toon_depth, line_width=2)
        creative_image.save(output_folder / f"{output_name}_creative.png")

    print(f"Saved images to: {output_folder.resolve()}")
    print(f"Contour edges found: {len(contour_edges)}")
    if creative:
        print("Creative output: manga-style hatching plus clean contours")


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def read_vector(text: str) -> np.ndarray:
    """Read a vector written like '1.0,-3.0,0.3' from the command line."""
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated numbers, e.g. 0,0,3")
    return np.array(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple NPR renderer for the skull coursework mesh")

    parser.add_argument("--mesh", required=True, help="Path to the OBJ/PLY/STL mesh file")
    parser.add_argument("--out", default="outputs/skull", help="Output prefix for the saved images")
    parser.add_argument("--creative", action="store_true", help="Also save the manga-style creative render")

    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--toon-levels", type=int, default=4)

    parser.add_argument("--eye", type=read_vector, default=read_vector("1.0,-3.0,0.3"))
    parser.add_argument("--target", type=read_vector, default=read_vector("0,0,0"))
    parser.add_argument("--up", type=read_vector, default=read_vector("0,0,1"))

    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    camera = Camera(
        eye=args.eye,
        target=args.target,
        up=args.up,
        width=args.width,
        height=args.height,
        fov_degrees=args.fov,
    )

    render_all_outputs(
        mesh=mesh,
        camera=camera,
        output_prefix=args.out,
        creative=args.creative,
        toon_levels=args.toon_levels,
    )


if __name__ == "__main__":
    main()
