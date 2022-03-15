import matplotlib.pyplot as plt

import numpy as np
from numpy.random import default_rng

from scipy.stats import uniform
from shapely import ops as ops
from shapely.geometry import shape, MultiLineString, Polygon, MultiPolygon


def realize_poisson_line_process(
    width=1, height=1, intensity=1, orientation=uniform(loc=0, scale=np.pi)
):
    """Draw a realization from a poisson line process
    inside a rectangle centered around (0,0) with given
    extent, intensity and orientation distributions.

    The pairs of distances of the lines to the origin and
    their orientations are drawn from a poisson process
    on [0,infty) x [0,2*pi).

    Parameters
    ----------
    width : float, optional
        The width of the rectangle, default is 1.0
    height : float, optional
        The height of the rectangle, default is 1.0
    intensity : float, optional
        The intensity of the poisson line process,
        default is 1.0.
    orientation : {`scipy.stats.rv_continuous`,
                   `scipy.stats.rv_discrete`}, optional
        The line orientation distribution, default
        is uniform distribution on [0, pi)

    Returns
    -------
    lines : list
        list of poisson lines which intersect the given
        rectangle. Each line is given as a pair of
        ``(slope, intercept)``.
    """

    rng = default_rng()

    # points within inner_radius are always inside the rectangle
    inner_radius = min(width / 2, height / 2)

    # points outside outer_radius are never inside the rectangle
    outer_radius = 0.5 * np.sqrt(width * width + height * height)

    lines = []

    distance = 0

    # radially simulate the distances between the lines and the origin
    while distance < outer_radius:
        distance -= np.log(rng.random()) / (2 * np.pi * intensity)

        # Draw alpha from the given orientation distribution.
        alpha = orientation.rvs()

        if rng.random() > 0.5:
            alpha = alpha + np.pi

        current_line = convert_line_representation(distance, alpha)

        # Check if lines between inner_radius and outer_radius intersect the rectangle
        if distance > inner_radius:
            if current_line[0] > 0:
                # Positive gradient => Check if lower right and upper left corner are on the same side of the line
                side_1 = side(current_line, (width / 2, -height / 2))
                side_2 = side(current_line, (-width / 2, height / 2))
            else:
                # Negative gradient => Check if upper right and lower left left corner are on the same side of the line
                side_1 = side(current_line, (width / 2, height / 2))
                side_2 = side(current_line, (-width / 2, -height / 2))
            if side_1 == side_2:
                continue

        lines.append(current_line)

    return lines


# Convert the representation of a line given by the distance to the origin and the angle to y=f(x)=m*x+t
def convert_line_representation(distance_to_origin, angle):
    m = -np.cos(angle) / np.sin(angle)
    x = distance_to_origin * np.cos(angle)
    y = distance_to_origin * np.sin(angle)
    t = y - m * x
    return m, t


# Determines on which side (relative to the directional vector (dx,dy) of the given line the point p lies.
# We assume that line is a tuple of m and t (f(x)=m*x+t)
def side(line, point):
    return np.cross(np.array((1, line[0])), point - np.array((0, line[1]))) > 0


def plot_lines(
    lines, low_x, high_x, low_y, high_y,
):
    for current_line in lines:
        y_min = current_line[0] * low_x + current_line[1]
        y_max = current_line[0] * high_x + current_line[1]
        plt.plot([low_x, high_x], [y_min, y_max])

    plt.xlim([low_x, high_x])
    plt.ylim([low_y, high_y])

    # plt.show()


def detect_shapes(
    lines, low_x, high_x, low_y, high_y,
):
    line_pairs = []
    for current_line in lines:
        y_min = current_line[0] * low_x + current_line[1]
        y_max = current_line[0] * high_x + current_line[1]
        line_pairs.append(((low_x, y_min), (high_x, y_max)))

    mls = MultiLineString(line_pairs)
    # Convert Lines to Polygons by applying a tiny buffer
    mls = mls.buffer(0.0000000000001)
    # Get outer boundary of the lines as a polygon
    boundary = mls.convex_hull
    # Get a difference to generate a multipolygon
    multipolygons = boundary.difference(mls)

    area = Polygon([[low_x, low_y], [high_x, low_y], [high_x, high_y], [low_x, high_y], [low_x, low_y]])
    if isinstance(multipolygons, Polygon):
        polygon = multipolygons.intersection(area)
        if polygon.is_empty:
            return []
        return [polygon]

    polygons = []
    for polygon in multipolygons.geoms:
        polygon = polygon.intersection(area)
        if polygon.is_empty:
            continue
        elif isinstance(polygon, Polygon):
            polygons.append(polygon)
        elif isinstance(polygon, MultiPolygon):
            polygons += list(polygon.geoms)

    return polygons


def run(width, height, intensity):
    orientation_distribution = uniform(loc=0, scale=np.pi)
    lines = realize_poisson_line_process(
        width, height, intensity, orientation_distribution
    )
    polygons = detect_shapes(lines, 0, width, 0, height)
    return (len(lines), len(polygons))


if __name__ == "__main__":
    width = 1
    height = 1
    intensity = 2
    orientation_distribution = uniform(loc=0, scale=np.pi)
    lines = realize_poisson_line_process(
        width, height, intensity, orientation_distribution
    )
    # If the orientation is uniformly distributed in [0,pi), the average number of lines intersecting a convex region
    # is given by the product of the intensity and the perimeter of the convex region.
    # Thus, we expect (width+height)*2*intensity lines crossing the rectangle in case of a uniform orientation distribution on [0,pi).

    print(len(lines), "lines")
    plot_lines(lines, 0, width, 0, height)

    polygons = detect_shapes(lines, 0, width, 0, height)
    print(len(polygons), "polygons")
    for polygon in polygons:
        plt.fill(*polygon.exterior.xy)
