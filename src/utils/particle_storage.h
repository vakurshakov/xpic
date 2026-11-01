#ifndef SRC_UTILS_PARTICLE_STORAGE_H
#define SRC_UTILS_PARTICLE_STORAGE_H

#include <petscdmswarm.h>

#include "src/pch.h"
#include "src/utils/world.h"

/**
 * @brief Particles of one sort as flat per-field arrays
 * (`DMSwarm` is only the field registry), without a per-cell order.
 *
 * Fields are registered by the sort between `initialize()` and
 * `commit_fields()`; the first one is the position `"r"`, which defines the
 * cell and the owner of a particle. A registered field can get a companion
 * `"<name>_prev"` that is not exchanged.
 *
 * Positions are periodically wrapped on periodic axes. On the other axes, the
 * particles that left the global domain are dropped by `update_cells()`.
 */
class ParticleStorage {
public:
  /// @brief Raw field arrays by name, valid until
  /// the next call that changes the number of particles.
  struct Fields {
    PetscInt size;
    std::map<std::string, PetscReal*> data;
    std::map<std::string, PetscInt> dof;  // Components of each array

    PetscReal* operator[](const std::string& name) const
    {
      return data.at(name);
    }
  };

  ParticleStorage(const World& world);

  PetscErrorCode initialize();
  PetscErrorCode finalize();

  PetscErrorCode register_field(const std::string& name, PetscInt dof);
  PetscErrorCode register_previous(const std::string& name);
  PetscErrorCode commit_fields();

  /// @brief Number of local particles.
  PetscErrorCode get_size(PetscInt* n) const;

  /**
   * @brief Adds `n` particles with undefined fields to the end of the arrays.
   * @param first Index of the first added particle.
   * @note Not collective. Invalidates the pointers of `Fields`.
   */
  PetscErrorCode grow(PetscInt n, PetscInt* first);

  /**
   * @brief Removes the particles with the given indices by replacing them with
   * the last ones. The order of the rest is lost.
   * @note Not collective. Invalidates the pointers of `Fields` and the order.
   */
  PetscErrorCode remove(std::vector<PetscInt> indices);

  /// @brief Removes all particles. The same notes as for `remove()`.
  PetscErrorCode clear();

  /**
   * @brief Gets the raw arrays. Call it once on the main thread and pass the
   * pointers to the parallel region. Getters write to different keys of `f`,
   * so they can be combined; one `restore_fields()` releases all of them and
   * must be called before anything that changes `size()`.
   *
   * `get_fields()` resets `f` and loads the exchanged fields only,
   * `get_previous()` adds the `"<name>_prev"` arrays,
   * `get_field()` adds exactly one array by its name (e.g. `"p"` or `"p_prev"`).
   */
  PetscErrorCode get_fields(Fields& f) const;
  PetscErrorCode get_previous(Fields& f) const;
  PetscErrorCode get_field(Fields& f, const std::string& name) const;
  PetscErrorCode restore_fields(Fields& f) const;

  /**
   * @brief Copies every field with a `_prev` companion to it.
   * @warning Particles are reordered by `update_cells()` and `sort()`,
   * so call this after them.
   */
  PetscErrorCode save_previous();

  /**
   * @brief Wraps the periodic coordinates and hands over the particles that
   * left the local box to the neighbours. Only they are touched: the order of
   * the others is kept, the removed ones are replaced by the last particles.
   * @note Particles are expected to move at most to the neighbour rank.
   */
  PetscErrorCode update_cells();

  /**
   * @brief Counting sort by the local cell. Cell `g` of `World::s_g()` is
   * `[offsets[g], offsets[g + 1])` of the field arrays. Previous fields
   * are not permuted.
   * @warning The order is stale after any change of positions or of `size()`.
   */
  PetscErrorCode sort();
  const std::vector<PetscInt>& cell_offsets() const;

private:
  struct Field {
    std::string name;
    PetscInt dof;
  };

  std::vector<PetscReal*> exchanged_arrays(const Fields& f) const;
  PetscInt cell(const PetscReal* r) const;
  PetscInt neighbor_slot(const PetscReal* r) const;
  void wrap(PetscReal* r) const;

  const World& world;
  DM swarm = NULL;
  std::vector<Field> fields;    // Exchanged between ranks
  std::vector<Field> previous;  // Fields that have the `_prev` companion
  PetscInt stride = 0;          // Components of all `fields` of a particle
  std::vector<PetscInt> offsets;
};

#endif  // SRC_UTILS_PARTICLE_STORAGE_H
