#include "parameters.h"

PetscReal __0th_order_spline(PetscReal s, Axis a) {
  if (Geom_n[a] == 1)
    return 1.0;

  s = std::abs(s);

  if (s <= 0.5) {
    return 1.0;
  }
  else { return 0.0; }
}

PetscReal __1st_order_spline(PetscReal s, Axis a) {
  if (Geom_n[a] == 1)
    return 1.0;

  s = std::abs(s);

  if (s <= 1.0) {
    return 1.0 - s;
  }
  else { return 0.0; }
}

PetscReal __2nd_order_spline(PetscReal s, Axis a) {
  if (Geom_n[a] == 1)
    return 1.0;

  s = std::abs(s);

  if (s <= 0.5) {
    return (0.75 - s * s);
  }
  else if (0.5 < s && s < 1.5) {
    return 0.5 * (1.5 - s) * (1.5 - s);
  }
  else { return 0.0; }
}

PetscReal __3rd_order_spline(PetscReal s, Axis a) {
  if (Geom_n[a] == 1)
    return 1.0;

  s = std::abs(s);
  PetscReal s2 = s * s;
  PetscReal s3 = s * s * s;

  if (s < 1.0) {
    return (4. - 6. * s2 + 3. * s3) / 6.;
  }
  else if (1.0 <= s && s < 2.0) {
    return (2. - s) * (2. - s) * (2. - s) / 6.;
  }
  else { return 0.0; }
}

PetscReal __4th_order_spline(PetscReal s, Axis a) {
  if (Geom_n[a] == 1)
    return 1.0;

  s = std::abs(s);
  PetscReal s2 = s * s;
  PetscReal s3 = s * s * s;
  PetscReal s4 = s * s * s * s;

  if (s <= 0.5) {
    return (115. / 192. - 5. / 8. * s2 + 1. / 4. * s4);
  }
  else if (0.5 < s && s <= 1.5) {
    return (55. + 20. * s - 120. * s2 + 80. * s3 - 16. * s4) / 96.;
  }
  else if (1.5 < s && s < 2.5) {
    return (5. - 2. * s) * (5. - 2. * s) * (5. - 2. * s) * (5. - 2. * s) / 384.;
  }
  else { return 0.0; }
}

PetscReal __5th_order_spline(PetscReal s, Axis a) {
  if (Geom_n[a] == 1)
    return 1.0;

  s  = std::abs(s);
  PetscReal s2 = s * s;
  PetscReal s3 = s * s * s;
  PetscReal s4 = s * s * s * s;
  PetscReal s5 = s * s * s * s * s;

  if (s <= 1.0) {
    return (11. / 20. - 0.5 * s2 + 0.25 * s4 - 1. / 12. * s5);
  }
  else if (1.0 < s && s <= 2.0) {
    return (17. / 40. + 5. / 8. * s - 7. / 4. * s2 + 5. / 4. * s3 - 3. / 8. * s4 + 1. / 24. * s5);
  }
  else if (2.0 < s && s < 3.0) {
    return (3. - s) * (3. - s) * (3. - s) * (3. - s) * (3. - s) / 120.;
  }
  else { return 0.0; }
}
