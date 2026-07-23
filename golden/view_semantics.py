"""View/reference-semantics parity test — run under both packages, compare output."""
import numpy as np
import tiny3d as t3d
g, u = t3d.geometry, t3d.utility

out = []
p = g.PointCloud()
p.points = u.Vector3dVector(np.arange(12.0).reshape(4, 3))

# np.asarray shares memory
a = np.asarray(p.points)
a[0, 0] = 42.0
out.append(("asarray_writethrough", p.points[0].tolist()))

# vector element assignment writes through
v = p.points
v[1] = [7.0, 8.0, 9.0]
out.append(("setitem_writethrough", np.asarray(p.points)[1].tolist()))

# append writes through
v.append([1.0, 2.0, 3.0])
out.append(("append_writethrough", len(p.points)))

# idiomatic bulk mutation
np.asarray(p.colors)  # empty view ok
p.colors = u.Vector3dVector(np.zeros((5, 3)))
np.asarray(p.colors)[:] = 0.25
out.append(("bulk_assign", np.asarray(p.colors).sum()))

# len/getitem through the live vector after external mutation
p2 = g.PointCloud()
p2.points = u.Vector3dVector(np.ones((3, 3)))
w = p2.points
p2.points = u.Vector3dVector(np.zeros((6, 3)))
out.append(("live_len_after_reassign", len(w)))

# mesh vectors
m = g.TriangleMesh()
m.vertices = u.Vector3dVector(np.arange(9.0).reshape(3, 3))
m.triangles = u.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
np.asarray(m.vertices)[2] = [9.0, 9.0, 9.0]
out.append(("mesh_vertex_writethrough", np.asarray(m.vertices)[2].tolist()))
np.asarray(m.triangles)[0, 0] = 2
out.append(("mesh_tri_writethrough", m.triangles[0].tolist()))

# free-standing vector: asarray shares with the vector object
fv = u.Vector3dVector(np.ones((2, 3)))
np.asarray(fv)[0, 0] = 5.0
out.append(("standalone_writethrough", fv[0].tolist()))

# copy is detached
import copy
cv = copy.copy(p.points)
cv[0] = [0.0, 0.0, 0.0]
out.append(("copy_detached", np.asarray(p.points)[0].tolist()))

# base object keeps cloud alive
def make():
    q = g.PointCloud()
    q.points = u.Vector3dVector(np.full((2, 3), 3.5))
    return np.asarray(q.points)
arr = make()
import gc; gc.collect()
out.append(("base_keepalive", arr.tolist()))

for k, val in out:
    print(k, val)
