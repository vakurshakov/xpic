#ifndef SRC_IMPLS_ECCAPFIM_CELL_TRAVERSAL_H
#define SRC_IMPLS_ECCAPFIM_CELL_TRAVERSAL_H

#include "src/pch.h"
#include "src/utils/vector3.h"

/**
 * @brief Finds a stopping points at cell edges along the
 * straight line between `start` and `end` points. On Yee
 * grid, edges, where electric field components are defined,
 * are shifted by a halfstep from nodes, obtained by `std::floor()`.
 *
 * @param[in] end   Last position of the particle.
 * @param[in] start Initial position of the particle.
 * @returns The sequence of start, intermediate stopping-points and the end point.
 *
 * @note The original implementation is taken from the repositories
 * https://github.com/francisengelmann/fast_voxel_traversal,
 * https://github.com/cgyurgyik/fast-voxel-traversal-algorithm.
 *
 * For a detailed explanation of the method see
 * Amanatides, John, and Andrew Woo. "A fast voxel traversal algorithm for ray tracing." Eurographics. Vol. 87. No. 3. 1987.
 */
std::vector<Vector3R> cell_traversal(const Vector3R& end, const Vector3R& start);

#endif  // SRC_IMPLS_ECCAPFIM_CELL_TRAVERSAL_H
