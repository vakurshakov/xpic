#include "particle_storage.h"

constexpr PetscInt slots = 27;
constexpr PetscInt center = 13;

// Packed particles, `stride` numbers each, grouped by the neighbour slot.
using Packed = std::vector<PetscReal>;

ParticleStorage::ParticleStorage(const World& world)
  : world(world)
{
}

PetscErrorCode ParticleStorage::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_WORLD, &swarm));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(DMSetDimension(swarm, 3));
  PetscCall(DMSwarmSetType(swarm, DMSWARM_BASIC));
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::finalize()
{
  PetscFunctionBeginUser;
  if (swarm)
    PetscCall(DMDestroy(&swarm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::register_field(
  const std::string& name, PetscInt dof)
{
  PetscFunctionBeginUser;
  PetscCheck(!fields.empty() || (name == "r" && dof == 3), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "The first field is the position \"r\" with 3 components");

  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, name.c_str(), dof, PETSC_REAL));
  fields.push_back({name, dof});
  stride += dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::register_previous(const std::string& name)
{
  PetscFunctionBeginUser;
  auto field = std::ranges::find(fields, name, &Field::name);
  PetscCheck(field != fields.end(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "Field \"%s\" is not registered to register previous from it", name.c_str());

  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, (name + "_prev").c_str(), field->dof, PETSC_REAL));
  previous.push_back(*field);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::commit_fields()
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSwarmSetLocalSizes(swarm, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::get_size(PetscInt* n) const
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(swarm, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::grow(PetscInt n, PetscInt* first)
{
  PetscFunctionBeginUser;
  PetscCall(get_size(first));
  PetscCall(DMSwarmAddNPoints(swarm, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::remove(std::vector<PetscInt> indices)
{
  PetscFunctionBeginUser;
  Fields f;
  PetscCall(get_fields(f));
  std::vector<PetscReal*> curr = exchanged_arrays(f);

  // Removing from the largest index keeps the last particle a non-removed one
  std::ranges::sort(indices, std::greater<>());
  PetscInt np = f.size;
  for (PetscInt i : indices) {
    np--;
    for (size_t k = 0; k < fields.size(); k++) {
      const PetscInt dof = fields[k].dof;
      if (i != np)
        std::copy_n(curr[k] + dof * np, dof, curr[k] + dof * i);
    }
  }
  PetscCall(restore_fields(f));
  PetscCall(DMSwarmSetLocalSizes(swarm, np, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::clear()
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmSetLocalSizes(swarm, 0, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::get_fields(Fields& f) const
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(swarm, &f.size));

  f.data.clear();
  for (const auto& field : fields)
    PetscCall(DMSwarmGetField(swarm, field.name.c_str(), NULL, NULL, (void**)&f.data[field.name]));
  for (const auto& field : previous)
    PetscCall(DMSwarmGetField(swarm, (field.name + "_prev").c_str(), NULL, NULL, (void**)&f.data[field.name + "_prev"]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::restore_fields(Fields& f) const
{
  PetscFunctionBeginUser;
  for (const auto& field : previous)
    PetscCall(DMSwarmRestoreField(swarm, (field.name + "_prev").c_str(), NULL, NULL, (void**)&f.data[field.name + "_prev"]));
  for (const auto& field : fields)
    PetscCall(DMSwarmRestoreField(swarm, field.name.c_str(), NULL, NULL, (void**)&f.data[field.name]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::save_previous()
{
  PetscFunctionBeginUser;
  Fields f;
  PetscCall(get_fields(f));
  for (const auto& [name, dof] : previous) {
    const PetscInt n = dof * f.size;
    const PetscReal* curr = f[name];
    PetscReal* prev = f[name + "_prev"];

#pragma omp parallel for
    for (PetscInt i = 0; i < n; i++)
      prev[i] = curr[i];
  }
  PetscCall(restore_fields(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The sender's slot `s` is the receiver's slot `26 - s`. With 2 ranks several
// slots share the same neighbour, so the messages are told apart by the tag.
PetscErrorCode ParticleStorage::update_cells()
{
  PetscFunctionBeginUser;
  Fields f;
  PetscCall(get_fields(f));
  std::vector<PetscReal*> curr = exchanged_arrays(f);

  // Threads own ascending chunks, so the indices are ascending in thread order
  std::vector<std::vector<PetscInt>> movers(omp_get_max_threads());
#pragma omp parallel
  {
    std::vector<PetscInt>& mine = movers[omp_get_thread_num()];

#pragma omp for schedule(static)
    for (PetscInt i = 0; i < f.size; i++) {
      PetscReal* r = &curr[0][3 * i];
      wrap(r);
      if (neighbor_slot(r) != center)
        mine.push_back(i);
    }
  }

  // Removing from the largest index keeps the last particle a non-mover
  Packed outgoing[slots];
  Packed incoming[slots];
  PetscInt np = f.size;
  for (auto vec = movers.rbegin(); vec != movers.rend(); vec++) {
    for (auto it = vec->rbegin(); it != vec->rend(); it++) {
      const PetscInt i = *it;
      Packed& out = outgoing[neighbor_slot(&curr[0][3 * i])];

      np--;
      for (size_t k = 0; k < fields.size(); k++) {
        const PetscInt dof = fields[k].dof;
        PetscReal* field = curr[k];
        out.insert(out.end(), field + dof * i, field + dof * (i + 1));

        if (i != np)
          std::copy_n(field + dof * np, dof, field + dof * i);
      }
    }
  }
  PetscCall(restore_fields(f));

  size_t o_num[slots];
  size_t i_num[slots];
  MPI_Request reqs[2 * slots];
  PetscMPIInt req = 0;

  for (PetscInt s = 0; s < slots; s++) {
    o_num[s] = outgoing[s].size();
    if (s == center)
      continue;
    PetscCallMPI(MPI_Isend(&o_num[s], 1, MPIU_SIZE_T, world.neighbors[s], 100 + s, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[s], 1, MPIU_SIZE_T, world.neighbors[s], 100 + slots - 1 - s, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  req = 0;
  for (PetscInt s = 0; s < slots; s++) {
    if (s == center)
      continue;
    incoming[s].resize(i_num[s]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), (PetscMPIInt)o_num[s], MPIU_REAL, world.neighbors[s], 200 + s, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[s].data(), (PetscMPIInt)i_num[s], MPIU_REAL, world.neighbors[s], 200 + slots - 1 - s, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  size_t received = 0;
  for (const Packed& in : incoming)
    received += in.size() / stride;
  PetscCall(DMSwarmSetLocalSizes(swarm, np + (PetscInt)received, -1));

  PetscCall(get_fields(f));
  curr = exchanged_arrays(f);
  PetscInt w = np;
  for (const Packed& in : incoming) {
    const PetscReal* src = in.data();
    for (size_t n = 0; n < in.size() / stride; n++, w++) {
      for (size_t k = 0; k < fields.size(); k++) {
        const PetscInt dof = fields[k].dof;
        std::copy_n(src, dof, curr[k] + dof * w);
        src += dof;
      }
    }
  }
  PetscCall(restore_fields(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::sort()
{
  PetscFunctionBeginUser;
  Fields f;
  PetscCall(get_fields(f));
  std::vector<PetscReal*> curr = exchanged_arrays(f);

  offsets.assign(world.size.elements_product() + 1, 0);
  std::vector<PetscInt> key(f.size);
  for (PetscInt i = 0; i < f.size; i++) {
    PetscAssert(neighbor_slot(&curr[0][3 * i]) == center, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      "Particle %d is outside of the local box, call update_cells() first", i);

    key[i] = cell(&curr[0][3 * i]);
    offsets[key[i] + 1]++;
  }

  for (size_t g = 0; g + 1 < offsets.size(); g++)
    offsets[g + 1] += offsets[g];

  std::vector<PetscInt> fill(offsets.begin(), offsets.end() - 1);
  std::vector<PetscInt> dst(f.size);
  for (PetscInt i = 0; i < f.size; i++)
    dst[i] = fill[key[i]]++;

  std::vector<PetscReal> scratch(stride * (size_t)f.size);
  for (size_t k = 0; k < fields.size(); k++) {
    const PetscInt dof = fields[k].dof;
    PetscReal* field = curr[k];

#pragma omp parallel for
    for (PetscInt i = 0; i < f.size; i++)
      std::copy_n(field + dof * i, dof, &scratch[dof * dst[i]]);

    std::copy_n(scratch.data(), dof * f.size, field);
  }
  PetscCall(restore_fields(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

const std::vector<PetscInt>& ParticleStorage::cell_offsets() const
{
  return offsets;
}

std::vector<PetscReal*> ParticleStorage::exchanged_arrays(const Fields& f) const
{
  std::vector<PetscReal*> arrays;
  for (const auto& field : fields)
    arrays.push_back(f[field.name]);
  return arrays;
}

PetscInt ParticleStorage::cell(const PetscReal* r) const
{
  return world.s_g(                         //
    FLOOR_STEP(r[X], dx) - world.start[X],  //
    FLOOR_STEP(r[Y], dy) - world.start[Y],  //
    FLOOR_STEP(r[Z], dz) - world.start[Z]);
}

// Layout of `World::neighbors`
PetscInt ParticleStorage::neighbor_slot(const PetscReal* r) const
{
  PetscInt slot = 0;
  for (PetscInt a : {Z, Y, X}) {
    const PetscInt c = FLOOR_STEP(r[a], Dx[a]);
    slot = 3 * slot + (c < world.start[a] ? 0 : (c < world.end[a] ? 1 : 2));
  }
  return slot;
}

void ParticleStorage::wrap(PetscReal* r) const
{
  for (PetscInt a : {X, Y, Z}) {
    if (world.bounds[a] != DM_BOUNDARY_PERIODIC)
      continue;

    r[a] = std::fmod(r[a], Geom[a]);
    if (r[a] < 0)
      r[a] += Geom[a];
    if (r[a] >= Geom[a])
      r[a] = 0;
  }
}
