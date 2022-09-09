import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
from time import time
from ufl import *
from dune.fem.view import adaptiveLeafGridView, geometryGridView
from dune.fem.space import dglagrange, combined
from dune.fem.scheme import galerkin
from dune.fem.plotting import plotPointData
from dune.mmesh import interfaceIndicator, normals, skeleton, trace, monolithicSolve

class Poroelastic:
  def __init__(self, grid, problem, igrid=None, dh=1, order=1):
    self.grid = grid
    self.problem = problem
    self.dh = dh
    self.dirichlet = problem.dirichlet
    self.uD = problem.uD
    self.pD = problem.pD
    self.pD_g = problem.pD_g
    u0 = problem.u0
    p0 = problem.p0
    p_g0 = problem.p_g0

    if igrid is None:
      self.igrid = adaptiveLeafGridView(grid.hierarchicalGrid.interfaceGrid)
    else:
      self.igrid = igrid

    # Bulk
    self.spaceu = dglagrange(grid, dimRange=problem.dim, order=order)
    self.spacep = dglagrange(grid, order=1)
    self.space = combined(self.spaceu, self.spacep)

    ux,  uy,  self.p  = TrialFunction(self.space)
    uux, uuy, self.pp = TestFunction(self.space)
    self.u  = as_vector([ux, uy])
    self.uu = as_vector([uux, uuy])

    self.sol = self.space.interpolate([u0[0], u0[1], p0], name="sol")
    self.solOld = self.sol.copy()

    self.uh = as_vector([self.sol[0], self.sol[1]])
    self.uhOld = as_vector([self.solOld[0], self.solOld[1]])
    self.ph = self.sol[2]
    self.phOld = self.solOld[2]

    # Interface
    self.ispace = dglagrange(self.igrid, order=1)

    self.p_g = TrialFunction(self.ispace)
    self.pp_g = TestFunction(self.ispace)

    self.ph_g = self.ispace.interpolate([p_g0], name="ph_g")
    self.ph_gOld = self.ph_g.copy()
    self.w = 0


  def __compile(self, verbose=False):
    I = interfaceIndicator(self.igrid)
    self.normal = normals(self.igrid)
    normal = self.normal

    P = self.problem
    x = SpatialCoordinate(self.space)
    n = FacetNormal(self.space)
    h = avg( CellVolume(self.space) ) / FacetArea(self.space)
    hBnd = CellVolume(self.space) / FacetArea(self.space)

    x_g = SpatialCoordinate(self.ispace)
    n_g = FacetNormal(self.ispace)
    h_g = avg( CellVolume(self.ispace) ) / FacetArea(self.ispace)
    h_gBnd = CellVolume(self.ispace) / FacetArea(self.ispace)

    # Coupling condition
    sol, solOld, uh, uhOld, ph, phOld = self.sol, self.solOld, self.uh, self.uhOld, self.ph, self.phOld
    ph_g, ph_gOld = self.ph_g, self.ph_gOld
    p_gamma = avg(skeleton(ph_g))
    trp = trace(sol)[2]

    # Use restricted traces that can be used in the interface dS integrals
    trSolP = trace(sol, restrictTo='+')
    trSolM = trace(sol, restrictTo='-')
    trSolOldP = trace(solOld, restrictTo='+')
    trSolOldM = trace(solOld, restrictTo='-')

    trup = as_vector([trSolP[0], trSolP[1]])
    trum = as_vector([trSolM[0], trSolM[1]])
    truOldp = as_vector([trSolOldP[0], trSolOldP[1]])
    truOldm = as_vector([trSolOldM[0], trSolOldM[1]])

    def C(p, p_gamma, n, omega):
        return - 2 * (P.K_n(n) / P.eta) / conditional(omega < 1e-3, 1e-3, omega) * (p_gamma - p)

    # Displacement
    betau = (P.lamb + 2 * P.mu) * P.beta0
    u, uu, p, pp = self.u, self.uu, self.p, self.pp

    sigma_tot = P.A(self.dh) * P.sigma_p(u) + P.sigma_m(u) - P.alpha * p * P.Id
    Au  = inner(sigma_tot, P.epsilon(uu)) * dx

    Au += betau / h * inner(jump(u), jump(uu)) * (1-I+P.k)*dS
    Au -= dot(dot(avg(sigma_tot), n('+')), jump(uu)) * (1-I+P.k)*dS

    Au += betau / hBnd * inner(u - as_vector(self.uD), uu) * self.dirichlet(x) * ds
    Au -= dot(dot(sigma_tot, n), uu) * self.dirichlet(x) * ds

#    Au += betau / hBnd * inner(dot(u, n), dot(uu, n)) * (1-self.dirichlet(x)) * ds
#    Au -= dot(dot(sigma_tot, n), n) * dot(uu, n) * (1-self.dirichlet(x)) * ds

    Au += inner((p_gamma - p_gamma**2/(2*P.M)) * n('+'), uu('+')) * I*dS
    Au += inner((p_gamma - p_gamma**2/(2*P.M)) * n('-'), uu('-')) * I*dS


    # Pressure
    betap = dot(dot(P.K, n('+')), n('+')) / P.eta * P.beta0
    betapBnd = dot(dot(P.K, n), n) / P.eta * P.beta0

    zeta = p / P.M + P.alpha * div(u)
    zetaOld = phOld / P.M + P.alpha * div(uhOld)
    Ap  = inner((zeta - zetaOld) / P.tau, pp) * dx
    Ap += inner(dot(P.K / P.eta, grad(p)), grad(pp)) * dx

    Ap += betap / h * inner(jump(p), jump(pp)) * (1-I)*dS
    Ap -= dot(dot(avg(dot(P.K / P.eta, grad(p))), n('+')), jump(pp)) * (1-I)*dS

    wb = P.omega(u('+'), u('-'), n('+'))
    Ap += C(p('+'), p_gamma, n('+'), wb) * pp('+') * I*dS
    Ap += C(p('-'), p_gamma, n('-'), wb) * pp('-') * I*dS

    Ap -= P.q(x) * pp * dx

    if self.pD is not None:
        Ap += betapBnd / hBnd * inner(p - self.pD, pp) * ds
        Ap -= dot(dot(dot(P.K / P.eta, grad(p)), n), pp) * ds


    # Fracture pressure
    p_g, pp_g = self.p_g, self.pp_g

    w = P.omega(trup, trum, normal)
    self.w = w
    whOld = P.omega(truOldp, truOldm, normal)
    betap_g = avg(w * P.K_tau(w)) / P.eta * P.beta0
    betap_gBnd = w * P.K_tau(w) / P.eta * P.beta0

    iAp  = inner(w * ( (p_g - ph_gOld) / P.M ) / P.tau, pp_g) * dx
    iAp += inner(w * P.K_tau(w) / P.eta * grad(p_g), grad(pp_g)) * dx

    iAp += (w - whOld) / P.tau * pp_g * dx

    iAp += betap_g / h_g * inner(jump(p_g), jump(pp_g)) * dS
    iAp -= inner(avg(w * P.K_tau(w) / P.eta * grad(p_g)), n_g('+')) * jump(pp_g) * dS

    iAp -= C(trp('+'), p_g, normal, w) * pp_g * dx
    iAp -= C(trp('-'), p_g, normal, w) * pp_g * dx

    iAp -= w * P.q(x_g) * pp_g * dx

    if self.pD_g is not None:
        iAp += betap_gBnd / h_gBnd * inner(p_g - self.pD_g, pp_g) * ds
        iAp -= w * P.K_tau(w) / P.eta * dot(grad(p_g), n_g) * pp_g * ds


    # Schemes
    self.scheme = galerkin([Au + Ap == 0], solver=("suitesparse", "umfpack"),
        parameters={"newton.verbose": verbose, "newton.tolerance": 1e-5}
    )
    self.ischeme = galerkin([iAp == 0], solver=("suitesparse", "umfpack"))


  def solve(self, verbose=False):
    if not hasattr(self, 'scheme'):
      self.__compile(verbose=verbose)

    start = time()

    def solve_():
      return monolithicSolve(
        schemes=(self.scheme, self.ischeme),
        targets=(self.sol, self.ph_g),
        tol=1e9, f_tol=1e-5, verbose=verbose, iter=10
      )

    converged = solve_()

    # If we have convergence issues...
    if not converged:
      tau = self.problem.tau
      count = 0
      # ... reduce time step size until it is convergent
      while not converged and count < 10:
        self.sol.assign(self.solOld)
        self.ph_g.assign(self.ph_gOld)
        tau.assign( tau.value / 2 )
        print(f"Retry with dt = {tau.value:.4e}")
        converged = solve_()
        self.write(count, prefix='iteration')
        count += 1

      # ... and increase again to match dt
      while tau.value < self.problem.dt - 1e-14:
        tau.assign( tau.value * 2 )
        print(f"Increase to dt = {tau.value:.4e}")
        converged = solve_()
        assert converged

    if verbose:
      print("Took {:.2f}s".format(time()-start))


  def proceed(self):
    self.solOld.assign(self.sol)
    self.ph_gOld.assign(self.ph_g)


  def write(self, step, directory='out', prefix=''):
    path = self.__getPath(directory, prefix, 'vtk')

    pointdata = {
        "u": [self.uh[0], self.uh[1], 0],
        "p": self.ph,
        "pOld": self.phOld,
        "uOld": self.uhOld,
        "d": self.dh
    }
    self.grid.writeVTK(path+'bulk-'+str(step), pointdata=pointdata, nonconforming=True)

    ipointdata = {
        "p": self.ph_g,
        "pOld": self.ph_gOld,
        "w": self.w
    }
    self.igrid.writeVTK(path+'fracture-'+str(step), pointdata=ipointdata, nonconforming=True)


  def plot(self, step, directory='out'):
    path = self.__getPath(directory, 'plot', 'png')

    def getClim(data):
      return [
        np.min([np.min(d.as_numpy) for d in data]),
        np.max([np.max(d.as_numpy) for d in data])
      ]

    # warp domain by displacement
    scale = 3
    x = SpatialCoordinate(self.space)
    uhInterp = self.spaceu.interpolate(x + self.uh * scale, name="uhInterp")
    geoGrid = geometryGridView(uhInterp)

    dim = self.problem.dim
    spacepgeo = dglagrange(geoGrid, order=1)
    spacedgeo = dglagrange(geoGrid, order=1)
    spaceugeo = dglagrange(geoGrid, order=2)
    spaceu1 = dglagrange(self.grid, order=2)

    ph0 = self.spacep.interpolate(self.ph, name="ph0")
    uh0x = spaceu1.interpolate(self.uh[0], name="uh0x")
    uh0y = spaceu1.interpolate(self.uh[1], name="uh0y")

    phgeo = spacepgeo.function(name="phgeo")
    dhgeo = spacedgeo.function(name="dhgeo")
    uhgeox = spaceugeo.function(name="uhgeox")
    uhgeoy = spaceugeo.function(name="uhgeoy")

    phgeo.as_numpy[:] = ph0.as_numpy
    dhgeo.as_numpy[:] = self.dh.as_numpy
    uhgeox.as_numpy[:] = uh0x.as_numpy
    uhgeoy.as_numpy[:] = uh0y.as_numpy

    def setLocatorAndFormatter(ax=plt.gca()):
      ax.xaxis.set_major_locator(MultipleLocator(1.0))
      ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
      ax.yaxis.set_major_locator(MultipleLocator(1.0))
      ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
      plt.tight_layout()

    def colorBar(clim, scientific=True):
      eps = 1e-2 * (clim[1] - clim[0])
      return {
        'ticks': np.linspace(clim[0] + eps, clim[1] - eps, 5, endpoint=True),
        'format': '%1.2e' if scientific else '%.2f',
        'extendfrac': 0
      }

    fig, ax = plt.subplots()
    clim = getClim([ph0, self.ph_g])
    plotPointData(ph0, figure=(fig, ax), gridLines=None, clim=clim, colorbar=colorBar(clim))
    plotPointData(self.ph_g, figure=(fig, ax), linewidth=0.01, colorbar=None, clim=clim)
    setLocatorAndFormatter(ax)
    plt.savefig(path+'pressure-'+str(step)+'.png', dpi=500, bbox_inches="tight")

    fig, ax = plt.subplots()
    clim = getClim([uhgeox])
    plotPointData(uhgeox, figure=(fig, ax), gridLines=None, colorbar=colorBar(clim))
    setLocatorAndFormatter(ax)
    plt.savefig(path+'displacementx-'+str(step)+'.png', dpi=500, bbox_inches="tight")

    fig, ax = plt.subplots()
    clim = getClim([uhgeoy])
    plotPointData(uhgeoy, figure=(fig, ax), gridLines=None, colorbar=colorBar(clim))
    setLocatorAndFormatter(ax)
    plt.savefig(path+'displacementy-'+str(step)+'.png', dpi=500, bbox_inches="tight")

    fig, ax = plt.subplots()
    plotPointData(dhgeo, figure=(fig, ax), gridLines=None, clim=[0,1], colorbar=colorBar([0,1], False))
    setLocatorAndFormatter(ax)
    plt.savefig(path+'phasefield-'+str(step)+'.png', dpi=500, bbox_inches="tight")
    plt.close('all')

  def __getPath(self, dir, prefix, type):
    name = self.problem.name
    directory = dir + '/' + name + '/' + type
    os.makedirs(directory, exist_ok=True)
    return directory + '/' + prefix + ('-' if prefix != '' else '')

