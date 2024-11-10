from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
import pandas as pd

brightness_threshold = 100
data = pd.read_csv("octree_data.csv")
data = data[data['value'] >= brightness_threshold]
print("Datapoints: ", len(data))

def create_box(row):
    x_start, x_end = row[0], row[1]
    y_start, y_end = row[2], row[3]
    z_start, z_end = row[4], row[5]
    grayscale_value = row[6]
    normalized_value = grayscale_value / 255.0

    center = [(x_start + x_end) / 2, (y_start + y_end) / 2, (z_start + z_end) / 2]
    extent = [x_end - x_start, y_end - y_start, z_end - z_start]
    box = pv.Cube(center=center, x_length=extent[0], y_length=extent[1], z_length=extent[2])
    color = (normalized_value, 0, 1 - normalized_value)
    return box, color

plotter = pv.Plotter()

with ThreadPoolExecutor() as executor:
    for box, color in executor.map(create_box, data.itertuples()):
        plotter.add_mesh(box, color=color)

plotter.show()

