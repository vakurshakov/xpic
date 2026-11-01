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

PetscErrorCode ParticleStorage::register_attribute(
  const std::string& name, PetscInt dof)
{
  PetscFunctionBeginUser;
  PetscCheck(!attributes.empty() || (name == "r" && dof == 3), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "The first attribute is the position \"r\" with 3 components");

  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, name.c_str(), dof, PETSC_REAL));
  attributes.push_back({name, dof});
  stride += dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::register_previous(const std::string& name)
{
  PetscFunctionBeginUser;
  auto attribute = std::ranges::find(attributes, name, &Attribute::name);
  PetscCheck(attribute != attributes.end(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "Attribute \"%s\" is not registered to register previous from it", name.c_str());

  PetscCall(DMSwarmRegisterPetscDatatypeField(swarm, (name + "_prev").c_str(), attribute->dof, PETSC_REAL));
  previous.push_back(*attribute);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::commit_attributes()
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  PetscCall(DMSwarmSetLocalSizes(swarm, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::get_attributes(Attributes& f) const
{
  PetscFunctionBeginUser;
  f.data.clear();
  f.dof.clear();
  for (const auto& attribute : attributes)
    PetscCall(get_attribute(f, attribute.name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::get_previous(Attributes& f) const
{
  PetscFunctionBeginUser;
  for (const auto& attribute : previous)
    PetscCall(get_attribute(f, attribute.name + "_prev"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::get_attribute(
  Attributes& f, const std::string& name) const
{
  PetscFunctionBeginUser;
  std::string base;
  const std::vector<Attribute>* list;

  if (!name.ends_with("_prev")) {
    base = name;
    list = &attributes;
  }
  else {
    base = name.substr(0, name.size() - 5);
    list = &previous;
  }

  auto attribute = std::ranges::find(*list, base, &Attribute::name);
  PetscCheck(attribute != list->end(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "Attribute \"%s\" is not registered", name.c_str());

  PetscCall(DMSwarmGetLocalSize(swarm, &f.size));
  PetscCall(DMSwarmGetField(swarm, name.c_str(), NULL, NULL, (void**)&f.data[name]));
  f.dof[name] = attribute->dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::restore_attributes(Attributes& f) const
{
  PetscFunctionBeginUser;
  for (auto& [name, array] : f.data)
    PetscCall(DMSwarmRestoreField(swarm, name.c_str(), NULL, NULL, (void**)&array));
  f.data.clear();
  f.dof.clear();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::save_previous()
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(f));
  PetscCall(get_previous(f));
  for (const auto& [name, dof] : previous) {
    const PetscInt n = dof * f.size;
    const PetscReal* curr = f[name];
    PetscReal* prev = f[name + "_prev"];

#pragma omp parallel for
    for (PetscInt i = 0; i < n; i++)
      prev[i] = curr[i];
  }
  PetscCall(restore_attributes(f));
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
  Attributes f;
  PetscCall(get_attributes(f));
  std::vector<PetscReal*> curr = exchanged_arrays(f);

  // Removing from the largest index keeps the last particle a non-removed one
  std::ranges::sort(indices, std::greater<>());
  PetscInt np = f.size;
  for (PetscInt i : indices) {
    np--;
    for (size_t k = 0; k < attributes.size(); k++) {
      const PetscInt dof = attributes[k].dof;
      if (i != np)
        std::copy_n(curr[k] + dof * np, dof, curr[k] + dof * i);
    }
  }
  PetscCall(restore_attributes(f));
  PetscCall(DMSwarmSetLocalSizes(swarm, np, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::clear()
{
  PetscFunctionBeginUser;
  PetscCall(DMSwarmSetLocalSizes(swarm, 0, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The sender's slot `s` is the receiver's slot `26 - s`. With 2 ranks several
// slots share the same neighbour, so the messages are told apart by the tag.
PetscErrorCode ParticleStorage::update_cells()
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(f));
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
      for (size_t k = 0; k < attributes.size(); k++) {
        const PetscInt dof = attributes[k].dof;
        PetscReal* attribute = curr[k];
        out.insert(out.end(), attribute + dof * i, attribute + dof * (i + 1));

        if (i != np)
          std::copy_n(attribute + dof * np, dof, attribute + dof * i);
      }
    }
  }
  PetscCall(restore_attributes(f));

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

  PetscCall(get_attributes(f));
  curr = exchanged_arrays(f);
  PetscInt w = np;
  for (const Packed& in : incoming) {
    const PetscReal* src = in.data();
    for (size_t n = 0; n < in.size() / stride; n++, w++) {
      for (size_t k = 0; k < attributes.size(); k++) {
        const PetscInt dof = attributes[k].dof;
        std::copy_n(src, dof, curr[k] + dof * w);
        src += dof;
      }
    }
  }
  PetscCall(restore_attributes(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ParticleStorage::sort()
{
  PetscFunctionBeginUser;
  Attributes f;
  PetscCall(get_attributes(f));
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
  for (size_t k = 0; k < attributes.size(); k++) {
    const PetscInt dof = attributes[k].dof;
    PetscReal* attribute = curr[k];

#pragma omp parallel for
    for (PetscInt i = 0; i < f.size; i++)
      std::copy_n(attribute + dof * i, dof, &scratch[dof * dst[i]]);

    std::copy_n(scratch.data(), dof * f.size, attribute);
  }
  PetscCall(restore_attributes(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

const std::vector<PetscInt>& ParticleStorage::cell_offsets() const
{
  return offsets;
}

std::vector<PetscReal*> ParticleStorage::exchanged_arrays(const Attributes& f) const
{
  std::vector<PetscReal*> arrays;
  for (const auto& attribute : attributes)
    arrays.push_back(f[attribute.name]);
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
