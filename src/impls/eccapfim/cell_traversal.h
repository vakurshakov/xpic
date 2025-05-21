#ifndef SRC_IMPLS_ECCAPFIM_CELL_TRAVERSAL_H
#define SRC_IMPLS_ECCAPFIM_CELL_TRAVERSAL_H

#include "src/pch.h"
#include "src/utils/vector3.h"

// https://github.com/francisengelmann/fast_voxel_traversal
// J. Amanatides, A. Woo. A Fast Voxel Traversal Algorithm for Ray Tracing. Eurographics '87
std::vector<Vector3R> cell_traversal(const Vector3R& end, const Vector3R& start);

#endif  // SRC_IMPLS_ECCAPFIM_CELL_TRAVERSAL_H
