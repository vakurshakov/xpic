Details
=======

Using PETSc SNES to solve equations of motion
---------------------------------------------

Setup of Jacobian-Free Newton Krylov iteration process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We would begin by recalling the Crank-Nicolson formulation of motion
equations. To move particle from timestep :math:`t^{n}` to step :math:`t^{n+1}`
one need to solve the system of nonlinear equations to find
:math:`\mathbf{x}^{n+1}`, :math:`\mathbf{v}^{n+1}`:

.. math:: \mathbf{x}^{n+1} = \mathbf{x}^{n} + \Delta t \mathbf{v}^{n+1/2},
  :label: cn-x

.. math:: \mathbf{v}^{n+1} = \mathbf{v}^{n} + \frac{\Delta t q}{m}\left(\mathbf{E}\left(\mathbf{x}^{n+1/2}\right) + \mathbf{v}^{n+1/2} \times \mathbf{B}\left(\mathbf{x}^{n+1/2}\right)\right),
  :label: cn-v

where
:math:`\Delta t` --- simulation timestep,
:math:`q, m` --- charge and mass of the particle;
half-step quantities are
:math:`\mathbf{x}^{n+1/2} = \left(\mathbf{x}^{n+1} + \mathbf{x}^{n}\right) / 2`,
:math:`\mathbf{v}^{n+1/2} = \left(\mathbf{v}^{n+1} + \mathbf{v}^{n}\right) / 2`
and fields are supposed to be interpolated onto particle's position at the
half-step. Further, the following notation for interpolated fields will be used
:math:`\mathbf{E}^{n+1/2} = \mathbf{E}\left(\mathbf{x}^{n+1/2}\right)`,
:math:`\mathbf{B}^{n+1/2} = \mathbf{B}\left(\mathbf{x}^{n+1/2}\right)`.
To solve this system using PETSc SNES, we should reformulate it in
the appropriate form by moving all quantities to the left hand side

.. math:: :label: cn-f

  \mathbf{F}(\mathbf{x}) = 0~\sim~\begin{pmatrix}
    \mathbf{x}^{n+1} - \mathbf{x}^{n} - \Delta t \mathbf{v}^{n+1/2} \\[0.1cm]
    \mathbf{v}^{n+1} - \mathbf{v}^{n} - \dfrac{\Delta t q}{m}\left(\mathbf{E}^{n+1/2} + \mathbf{v}^{n+1/2} \times \mathbf{B}^{n+1/2}\right)
  \end{pmatrix} = 0.

The above is the vector with six components, where PETSc iteration vector
:math:`\mathbf{x} = (\mathbf{x}^{n+1}, \mathbf{v}^{n+1}`).
To complete the default iteration process, we need to find the Jacobian
matrix. It can be easily calculated for the given iteration function:

.. math:: :label: cn-jac

  \mathbf{J}(\mathbf{x}) = \partial \mathbf{F} / \partial \mathbf{x} = \begin{pmatrix}
    1 & 0 & 0 & -\Delta t/2 & 0 & 0 \\[0.1cm]
    0 & 1 & 0 & 0 & -\Delta t/2 & 0 \\[0.1cm]
    0 & 0 & 1 & 0 & 0 & -\Delta t/2 \\[0.1cm]
    \partial_x A_x & \partial_y A_x & \partial_z A_x & 1 & -\alpha B_z & +\alpha B_y \\[0.1cm]
    \partial_x A_y & \partial_y A_y & \partial_z A_y & +\alpha B_z & 1 & -\alpha B_x \\[0.1cm]
    \partial_x A_z & \partial_y A_z & \partial_z A_z & -\alpha B_y & +\alpha B_x & 1 \\[0.1cm]
  \end{pmatrix}\!,

here :math:`\alpha = q \Delta t / 2 m` and :math:`\mathbf{A} = - \alpha \left(2 \mathbf{E}^{n+1/2} + \mathbf{v}^{n} \times \mathbf{B}^{n+1/2}\right)`.
By setting both iteration function and Jacobian with `SNESSetFunction() <https://petsc.org/main/manualpages/SNES/SNESSetFunction/>`_
and `SNESSetJacobian() <https://petsc.org/main/manualpages/SNES/SNESSetJacobian/>`_,
most of the PETSc features to solve this system of equations is enabled.

However, the storage restrictions on explicitly calculated field
gradients or inability to calculate the Jacobian at all, in cases
where some additional force is added into :math:`\mathbf{F}(\mathbf{x})`,
favors the different approach called Jacobian-Free Newton Krylov (JFNK)
method. The idea behind this method is to use the directional difference
instead of the Jacobian-vector product:

.. math:: \left(\mathbf{a}, \partial \mathbf{F} / \partial \mathbf{x}\right) \approx \left(\mathbf{F}(\mathbf{x} + h \mathbf{a}) - \mathbf{F}(\mathbf{x}) \right) \! / h.
  :label: jfnk-diff

For the rest of the implementation details we currently refer to
`SNES: Nonlinear Solvers, Matrix-free methods <https://petsc.org/main/manual/snes/#matrix-free-methods>`_.

Setup of nonlinear Richardson (Picard) iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simpler iteration process can be constructed from the above Crank-Nicolson
scheme: :eq:`cn-x`, :eq:`cn-v` by analytically inverting the second
equation. If we denote the iteration step of a nonlinear solver by
:math:`k`, the Picard iteration process will look like this

.. math:: \mathbf{a}^k = \mathbf{v}^{n} + \alpha \mathbf{E}^{n+1/2,\, k},
  :label: picard-a

.. math:: \mathbf{v}^{n+1/2,\, k+1} = \frac{\mathbf{a}^k + \alpha \mathbf{a}^k \times \mathbf{B}^{n+1/2,\, k}  + \alpha^2 (\mathbf{a}^k, \mathbf{B}^{n+1/2,\, k}) \mathbf{B}^{n+1/2,\, k}}{1 + (\alpha B^{n+1/2,\, k})^2}\,,
  :label: picard-vh

.. math:: \mathbf{x}^{n+1,\, k+1} = \mathbf{x}^{n} + \Delta t \mathbf{v}^{n+1/2,\, k+1},
  :label: picard-x

.. math:: \mathbf{v}^{n+1,\, k+1} = 2 \mathbf{v}^{n+1/2,\, k+1} - \mathbf{v}^{n}.
  :label: picard-v

This is the system of the same six variables to be iterated by SNES.
To setup the iteration process, we should explicitly set the type of
SNES instance to be `SNESNRICHARDSON <https://petsc.org/main/manualpages/SNES/SNESNRICHARDSON/>`_
and use `SNESSetFunction() <https://petsc.org/main/manualpages/SNES/SNESSetFunction/>`_.
However, there is a pitfall in a realization of the iteration function.
Using the simple algorithm described above, one would expect

.. math:: \mathbf{x}^{k+1} = \mathbf{f}(\mathbf{x}^{k})
  :label: picard-expectation

with :math:`\mathbf{f}(\mathbf{x})` being the function that implements :eq:`picard-a` -- :eq:`picard-v`.
Nevertheless, PETSc nonlinear Richardson iteration process has the following form:

.. math:: \mathbf{x}^{k+1} = \mathbf{x}^{k} - \lambda \tilde{\mathbf{F}}(\mathbf{x}^{k}),
  :label: picard-petsc

where :math:`\lambda \in (0, 1]` --- damping coefficient that can be
controlled by `SNESLineSearchSetDamping() <https://petsc.org/main/manualpages/SNES/SNESLineSearchSetDamping/>`_
and :math:`\lambda = 1` by default (no damping), :math:`\tilde{\mathbf{F}}(\mathbf{x})`
--- iteration function to be set.  Thus, to achieve a correct iteration
process, one must use

.. math:: \tilde{\mathbf{F}}(\mathbf{x}) = \mathbf{x} - \mathbf{f}(\mathbf{x}).
  :label: picard-correct

