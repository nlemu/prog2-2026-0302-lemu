import csv
import numpy as np
from pykdtree.kdtree import KDTree

points = np.load("points.npy")
meta = np.load("meta.npy", allow_pickle=True)

q_array = np.loadtxt("query.csv", delimiter=",", skiprows=1)

points = np.asarray(points, dtype=np.float64)
q_array = np.asarray(q_array, dtype=np.float64)

tree = KDTree(points)

_, idxs = tree.query(q_array, k=3)

out = meta[idxs].reshape(len(q_array), 6)

header = [
    "top1_title", "top1_id",
    "top2_title", "top2_id",
    "top3_title", "top3_id",
]

with open("out.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(out)
