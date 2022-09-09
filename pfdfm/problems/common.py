from ufl import *
from dune.ufl import Constant
from dune.fem.function import integrate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=int, default=1, help='Number of problem (1: tip, 2: joining)')
parser.add_argument('--hf', type=float, default=0.02, help='Mesh size at fracture')
args = parser.parse_args()

class Common:
  def __init__(self, problem=args.problem, h=0.2, hf=args.hf, dt=0.1, T=10):
    self.name = "common"
    self.number = problem
    self.dim = 2
    self.domainSize = 4
    self.h = h
    self.hf = hf
    self.threshold = 1e-2

    self.dt = dt
    self.T = T

    Id = Identity(self.dim)
    self.Id = Id
    E  = 1e8
    nu = 0.2
    self.lamb    = Constant( E*nu/((1+nu)*(1-2*nu)), name="lambda")
    self.mu      = Constant( E/(2*(1+nu)), name="mu")
    self.M       = Constant(   1e8, name="M")
    self.alpha   = Constant(     1, name="alpha")
    self.eta     = Constant(  1e-3, name="viscosity")
    self.K0      = Constant( 1e-12, name="K0")
    self.K       = self.K0 * Id
    self.g_c     = Constant(   1e3, name="g_c")
    self.omega0  = Constant( 1e-14, name="omega0")
    self.beta0   = Constant(   1e2, name="beta0")
    self.k       = Constant( 1e-12, name="k")
    self.p_inj   = Constant(  5e-3, name="p_inj")
    self.d0      = Constant(   1e3, name="d0")
    self.l       = Constant( 2*self.hf, name="l")

    if problem == 1:
      import pfdfm.grids.tip as meshfile
      self.name = "horizontal-"+str(hf)
    elif problem == 2:
      import pfdfm.grids.joining as meshfile
      self.name = "joining"
      self.T = 25
    else:
      raise Exception("Problem " + str(problem) + " not known!")

    self.meshfile = meshfile
    self.mshfile = meshfile.create(h=self.h, hf=self.hf, l=self.domainSize)
    self.tau = Constant(self.dt, name="tau")

    self.epsilon = lambda u: 0.5 * (grad(u) + grad(u).T)
    tr_p = lambda u: conditional(div(u) > 0, div(u), 0)
    tr_m = lambda u: conditional(div(u) < 0, div(u), 0)
    self.sigma_p = lambda u: (2 / self.dim * self.mu + self.lamb) * tr_p(u) * Id + 2 * self.mu * (self.epsilon(u) - 1 / self.dim * div(u) * Id)
    self.sigma_m = lambda u: (2 / self.dim * self.mu + self.lamb) * tr_m(u) * Id
    self.Psi_p = lambda u: 0.5 * inner(self.sigma_p(u), self.epsilon(u))
    self.Psi_m = lambda u: 0.5 * inner(self.sigma_m(u), self.epsilon(u))

    self.A = lambda d: (1-self.k) * d**2 + self.k
    self.dA = lambda d: 2 * (1-self.k) * d

    self.omega = lambda up, um, n: conditional(dot(um, n) - dot(up, n) > 0, dot(um, n) - dot(up, n), 0) + self.omega0
    self.K_tau = lambda w: w**2 / 12
    self.K_n   = lambda n: dot(n, dot(self.K, n))

    pointsource = lambda x0: (lambda x: self.p_inj * self.d0 / 3.14 * exp( -self.d0 * dot(x-as_vector(x0), x-as_vector(x0)) ))
    self.q = lambda x: pointsource([2,2])(x)
    self.Q_ = None

    l = self.domainSize
    self.dirichlet = lambda x: \
      conditional(x[0] < 1e-6, 1, 0) + \
      conditional(x[0] > l-1e-6, 1, 0) + \
      conditional(x[1] < 1e-6, 1, 0) + \
      conditional(x[1] > l-1e-6, 1, 0)
    self.uD = [0,0]
    self.pD = None
    self.pD_g = None
    self.u0 = [0,0]
    self.p0 = 0
    self.p_g0 = 0

  def W(self, model, dh):
    uh, ph, ph_g, w = model.uh, model.ph, model.ph_g, model.w
    gammal = lambda dh: 1/(2*self.l) * (1-dh)**2 + self.l/2 * dot(grad(dh), grad(dh))
    W = integrate(model.grid, self.A(dh) * self.Psi_p(uh) + self.Psi_m(uh) + ph**2/(2*self.M) + self.g_c * gammal(dh), order=2)
    W += integrate(model.igrid, w * ph_g**2/(2*self.M) + self.g_c, order=2)
    return W

  def Q(self, model, proceed=False):
    ph, ph_g, w = model.ph, model.ph_g, model.w
    x = SpatialCoordinate(model.space)
    x_g = SpatialCoordinate(model.ispace)

    if self.Q_ is None:
      self.Q_ = integrate(model.igrid, self.g_c, order=0)

    Q1 = self.Q_
    Q1 += self.dt * integrate(model.grid, ph * self.q(x), order=5)
    Q1 += self.dt * integrate(model.igrid, ph_g * w * self.q(x_g), order=5)

    if proceed:
      self.Q_ = Q1

    return Q1
