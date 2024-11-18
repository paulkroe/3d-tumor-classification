# conda install -c conda-forge pandas pyvista vtk
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libsqlite3.so.0
from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
import pandas as pd
import numpy as np
import ctypes
import h5py
import argparse
lib = ctypes.CDLL('./librenderer.so')

parser = argparse.ArgumentParser()

parser.add_argument("vol_nr", type=int, help="Volume Number")  
parser.add_argument("max_depth", type=int, help="Max Depth")  
parser.add_argument("threshold", type=int, help="Threshold")  

def open_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        pred = f['pred'][()]
        label = f['mask'][()]
    return pred, label

args = parser.parse_args()

def c_call(input, filename):
    input_flattened = input.flatten().astype(np.int32)
    c_array = input_flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    height, width, depth = input.shape
    c_filename = ctypes.c_char_p(filename.encode('utf-8'))
    lib.process_volume(c_array, height, width, depth, args.max_depth, args.threshold, c_filename)


file_path = f"../../preds/content/preds/volume_{args.vol_nr}"
pred, label = open_h5_file(file_path)

c_call(pred, "pred.csv")
c_call(label, "label.csv")

label = pd.read_csv("label.csv")
label = label[label['value'] == 1]
print("Label Datapoints: ", len(label))

pred = pd.read_csv("pred.csv")
pred = pred[pred['value'] == 1]
print("pred Datapoints: ", len(pred))
print("Legend: Green - True Positive, Red - False Negative, Blue - False Positive")

data = pd.merge(label, pred, on=['x_start', 'x_end', 'y_start', 'y_end', 'z_start', 'z_end'], 
                suffixes=('_label', '_pred'), how='outer').fillna(0)


def create_box(row):
    x_start, x_end = row[1], row[2]
    y_start, y_end = row[3], row[4]
    z_start, z_end = row[5], row[6]
    
    if row[7] == 1 and row[8] == 1:
        color = "green"
    elif row[7] == 1:
        color = "red"
    elif row[8] == 1:
        color = "blue"

    center = [(x_start + x_end) / 2, (y_start + y_end) / 2, (z_start + z_end) / 2]
    extent = [x_end - x_start, y_end - y_start, z_end - z_start]
    box = pv.Cube(center=center, x_length=extent[0], y_length=extent[1], z_length=extent[2])
    return box, color


plotter = pv.Plotter()

with ThreadPoolExecutor() as executor:
    for box, color in executor.map(create_box, data.itertuples()):
        plotter.add_mesh(box, color=color)

plotter.show()

