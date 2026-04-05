import struct
import numpy as np

path = "../Release/test_idx/centers"

with open(path, "rb") as f:
    rows = struct.unpack("<Q", f.read(8))[0]
    dim = struct.unpack("<I", f.read(4))[0]
    data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, dim)

print("rows =", rows)
print("dim  =", dim)
print(data)