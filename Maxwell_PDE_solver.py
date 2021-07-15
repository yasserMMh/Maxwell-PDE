from dolfin import *
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
import pandas as pd
import scipy.constants
import inspect
from inspect import signature

# Elliptical2D class is used to solve elliptical 2D Maxwell PDE
# -div(1/mu grad(A_z)) = J_z
# mu = mu_0 * mu_r => permeability
# mu_r could be found in the following link "https://www.engineeringtoolbox.com/permeability-d_1923.html"
# A_z => magnetic potential over z axis
# J_z => current density over z axis


class Elliptical2D:
    mu0 = scipy.constants.mu_0

    def __init__(self, mu_r=1, source='-6', degree=1):
        self.mu = mu_r * Elliptical2D.mu0
        self.J_z = Constant(source)

    ###############################################################################
    @staticmethod
    def mu(A_z):
        "Return nonlinear coefficient"
        return 1 + A_z ** 2
    ###############################################################################
    @staticmethod
    def source_Nonlinear(mu=mu.__func__):
        x, y = sym.symbols('x[0], x[1]')
        A_z = 1 + x + 2 * y
        J_z = - sym.diff(mu(A_z) * sym.diff(A_z, x), x) - sym.diff(mu(A_z) * sym.diff(A_z, y), y)
        J_z = sym.simplify(J_z)
        A_z_code = sym.printing.ccode(A_z)
        J_z_code = sym.printing.ccode(J_z)
        print('A_z =', A_z_code)
        print('J_z =', J_z_code)
        return J_z_code, A_z_code

    # Define Mesh
    @staticmethod
    def create_RectangleMesh(x1=0, y1=0, x2=1, y2=1, nx=8, ny=8):
        start_point = Point(x1, y1)
        end_point = Point(x2, y2)
        mesh = RectangleMesh(start_point, end_point, nx, ny)
        return mesh
    ###############################################################################
    @staticmethod
    def create_UnitMesh(nx=8, ny=8):
        mesh = UnitSquareMesh(nx, ny)
        return mesh
    ###############################################################################
    # Create Space and discretization
    @staticmethod
    def discretized_space(mesh, elementtype='Lagrange', degree=1):
        e = FiniteElement(elementtype, triangle, degree)
        space = FunctionSpace(mesh, elementtype, degree)
        return space
    ###############################################################################
    # Define Boundary
    @staticmethod
    def boundary(x, on_boundary):
        return on_boundary
    ###############################################################################
    # Define Boundary Conditions
    @staticmethod
    def boundary_conditions(space, boundary=boundary.__func__, u_d='0', degree=1):
        u_d = Expression(u_d, degree=degree)
        bc = DirichletBC(space, u_d, boundary)
        return bc
    ###############################################################################
    # Define Trial & Test functions
    @staticmethod
    def test_and_trail_fun_linear(space):
        w = TestFunction(space)
        A_z = TrialFunction(space)
        return w, A_z
    ###############################################################################
    @staticmethod
    def test_and_trail_fun_Nonlinear(space, J_z_code, degree=2):
        w = TestFunction(space)
        A_z = Function(space)  # Note: not TrialFunction!
        J_z = Expression(J_z_code, degree=degree)
        return w, A_z, J_z
    ###############################################################################
    # Define system equation
    def system_eq_linear(self, w, A_z, mesh):
        dx = Measure('dx', domain=mesh)
        a = 1 / self.mu * dot(grad(w), grad(A_z)) * dx
        L = self.J_z * w * dx
        return a, L
    ###############################################################################
    @staticmethod
    def solve(space, a, L, bc, p=True):
        A_z = Function(space)
        solve(a == L, A_z, bc)
        if p:
            plot(A_z)
            # plot(mesh)
            plt.show()
        return A_z
    ###############################################################################
    @staticmethod
    def system_eq_Nonlinear(w, A_z, J_z, mu=mu.__func__):
        F = mu(A_z)*dot(grad(A_z), grad(w))*dx - J_z*w*dx
        return F
    ###############################################################################
    @staticmethod
    def solve_Nonlinear(F, A_z, bc, p=True):
        solve(F == 0, A_z, bc)
        if p:
            plot(A_z)
            plt.show()
        return A_z
    ###############################################################################


# def solve_elliptic():
#     instance = Elliptical2D(1, '-6', 0)
#     instance.mu = 1
#     mesh = instance.create_mesh()
#     space = instance.discretized_space(mesh)
#     w, A_z = instance.test_and_trail_fun(space)
#     bc = instance.boundary_conditions(space)
#     a, L = instance.system_eq(w, A_z, mesh)
#     instance.solve(space, a, L, bc)
#
#
# solve_elliptic()
def mu(A_z):
    "Return nonlinear coefficient"
    return 1 + A_z ** 2
instance = Elliptical2D()
mesh = instance.create_UnitMesh()
space = instance.discretized_space(mesh)
J_z_code, A_z_code = instance.source_Nonlinear(mu)
bc = instance.boundary_conditions(space, u_d=A_z_code, degree=2)
w, A_z, J_z= instance.test_and_trail_fun_Nonlinear(space, J_z_code)
F = instance.system_eq_Nonlinear(w, A_z, J_z)
instance.solve_Nonlinear(F, A_z, bc)



class Parabolic2D:
    mu0 = scipy.constants.mu_0
    rho = -1
    period = 1
    steps = 10
    dt = period / steps  # step size

    def __init__(self, mu_r=1, source='-6', degree=1):
        self.mu = mu_r * Elliptical2D.mu0
        self.J_z = Constant(source)
    ###############################################################################
    # Define Mesh
    @staticmethod
    def create_RectangleMesh(x1=0, y1=0, x2=1, y2=1, nx=8, ny=8):
        start_point = Point(x1, y1)
        end_point = Point(x2, y2)
        mesh = RectangleMesh(start_point, end_point, nx, ny)
        return mesh
    ###############################################################################
    @staticmethod
    def create_UnitMesh(nx=8, ny=8):
        mesh = UnitSquareMesh(nx, ny)
        return mesh
    ###############################################################################
    # Create Space and discretization
    @staticmethod
    def discretized_space(mesh, elementtype='Lagrange', degree=1):
        e = FiniteElement(elementtype, triangle, degree)
        space = FunctionSpace(mesh, elementtype, degree)
        return space
    ###############################################################################
    # Define Boundary
    @staticmethod
    def boundary(x, on_boundary):
        return on_boundary
    ###############################################################################
    # Define Boundary Conditions
    @staticmethod
    def boundary_conditions(space, boundary=boundary.__func__, A_z_d='0', degree=1):
        # A_z_d = Expression(A_z_d, degree=degree)
        bc = DirichletBC(space, A_z_d, boundary)
        return bc
    ###############################################################################
    # Define initial value
    @staticmethod
    def initial_conditions(space, A_z_d='0'):
        A_z_n = interpolate(A_z_d, space)
        # A_z_n = project(A_z_d, space)
        return A_z_n
    # Define Trial & Test functions
    @staticmethod
    def test_and_trail_fun(space):
        w = TestFunction(space)
        A_z = TrialFunction(space)
        return w, A_z
    ###############################################################################
    # Define system equation
    def system_eq(self, w, A_z, v, A_z_n, mesh):
        dx = Measure('dx', domain=mesh)
        dt = Parabolic2D.dt
        rho = Parabolic2D.rho
        a = 1 / self.mu * dt * dot(grad(w), grad(A_z)) * dx - w * rho * A_z * dx #+ w * rho * grad(v) * dx
        L = dt * self.J_z * w * dx - w * rho * A_z_n * dx
        return a, L
    ###############################################################################
    @staticmethod
    def solve(space, a, L, A_z_d, A_z_n, bc, p=True):
        A_z = Function(space)
        t = 0
        for n in range(Parabolic2D.steps):
            t += Parabolic2D.dt
            A_z_d.t = t
            solve(a == L, A_z, bc)
            plot(A_z)
            A_z_n.assign(A_z)
        if p:
            plot(A_z)
            plt.show()
        return A_z
    ###############################################################################


# instance = Parabolic2D()
# instance.J_z = Constant(1.2 - 2 - 2*3)
# instance.mu = 1
# mesh = instance.create_UnitMesh()
# space = instance.discretized_space(mesh)
# A_z_D = Expression('1 + x[0]*x[0] + 3*x[1]*x[1] + 1.2*t', degree=2, t=0)
# bc = instance.boundary_conditions(space, A_z_d=A_z_D)
# A_z_n = instance.initial_conditions(space, A_z_D)
# w, A_z = instance.test_and_trail_fun(space)
# a, L = instance.system_eq(w, A_z, 1, A_z_n, mesh)
# instance.solve(space, a, L, A_z_D, A_z_n, bc)
