Details
=======

Using PETSc SNES to solve equations of motion
---------------------------------------------

We would begin by recalling the Crank-Nicolson formulation of motion
equations. To move particle from timestep :math:`t^{n}` to step :math:`t^{n+1}`
one need to solve the system of nonlinear equations to find
:math:`\mathbf{x}^{n+1}`, :math:`\mathbf{v}^{n+1}`:

.. math:: \mathbf{x}^{n+1} = \mathbf{x}^{n} + \Delta t \mathbf{v}^{n+1/2},

.. math:: \mathbf{v}^{n+1} = \mathbf{v}^{n} + \frac{\Delta t q}{m}\left(\mathbf{E}\left(\mathbf{x}^{n+1/2}\right) + \mathbf{v}^{n+1/2} \times \mathbf{B}\left(\mathbf{x}^{n+1/2}\right)\right),

where quantities at the half-step are
:math:`\mathbf{x}^{n+1/2} = \left(\mathbf{x}^{n+1} + \mathbf{x}^{n}\right) / 2`,
:math:`\mathbf{v}^{n+1/2} = \left(\mathbf{v}^{n+1} + \mathbf{v}^{n}\right) / 2`
and fields are supposed to be interpolated onto particle's position.
To solve this system using PETSc SNES, we should reformulate it in
the appropriate form by moving all quantities to the left hand side

.. math::

  \mathbf{F}(\mathbf{x}) = 0~\sim~\begin{pmatrix}
    \mathbf{x}^{n+1} - \mathbf{x}^{n} - \Delta t \mathbf{v}^{n+1/2} \\[0.4cm]
    \mathbf{v}^{n+1} - \mathbf{v}^{n} - \dfrac{\Delta t q}{m}\left(\mathbf{E}\left(\mathbf{x}^{n+1/2}\right) + \mathbf{v}^{n+1/2} \times \mathbf{B}\left(\mathbf{x}^{n+1/2}\right)\right)
  \end{pmatrix} = 0.

The above is the vector with six components where PETSc iteration vector
:math:`\mathbf{x} = (\mathbf{x}^{n+1}, \mathbf{v}^{n+1}`).
To complete the default iteration process, we need to find the Jacobian
matrix. It can be easily calculated for the given iteration function:

.. math::

  \mathbf{J}(\mathbf{x}) = \partial \mathbf{F} / \partial \mathbf{x} = \begin{pmatrix}
    1 & 0 & 0 & -\Delta t/2 & 0 & 0 \\[0.1cm]
    0 & 1 & 0 & 0 & -\Delta t/2 & 0 \\[0.1cm]
    0 & 0 & 1 & 0 & 0 & -\Delta t/2 \\[0.1cm]
    \partial_x A_x & \partial_y A_x & \partial_z A_x & 1 & -\alpha B_z & +\alpha B_y \\[0.1cm]
    \partial_x A_y & \partial_y A_y & \partial_z A_y & +\alpha B_z & 1 & -\alpha B_x \\[0.1cm]
    \partial_x A_z & \partial_y A_z & \partial_z A_z & -\alpha B_y & +\alpha B_x & 1 \\[0.1cm]
  \end{pmatrix}\!,

here :math:`\alpha = q \Delta t / 2 m` and :math:`\mathbf{A} = - \alpha \left(2 \mathbf{E}\left(\mathbf{x}^{n+1/2}\right) + \mathbf{v}^{n} \times \mathbf{B}\left(\mathbf{x}^{n+1/2}\right)\right)`.
Setting both iteration function and Jacobian with `SNESSetFunction() <https://petsc.org/main/manualpages/SNES/SNESSetFunction/>`_
and `SNESSetJacobian() <https://petsc.org/main/manualpages/SNES/SNESSetJacobian/>`_
enables most of the PETSc features to solve this system of equations.

However, the storage restrictions on explicitly calculated field
gradients or inability to calculate the Jacobian at all, in cases
where some additional force is added into :math:`\mathbf{F}(\mathbf{x})`,
favors the different approach called Jacobian-Free Newton Krylov (JFNK)
method. The idea behind this method is to use the directional difference
instead of the Jacobian-vector product:

.. math:: \left(\mathbf{a}, \partial \mathbf{F} / \partial \mathbf{x}\right) \approx \left(\mathbf{F}(\mathbf{x} + h \mathbf{a}) - \mathbf{F}(\mathbf{x}) \right) \! / h.

For the rest of the implementation details we currently refer to
`SNES: Nonlinear Solvers, Matrix-free methods <https://petsc.org/main/manual/snes/#matrix-free-methods>`_.

