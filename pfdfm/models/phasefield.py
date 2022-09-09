import os
from time import time
from ufl import *
from dune.fem.space import dglagrange
from dune.fem.scheme import galerkin
from dune.fem.plotting import plotPointData
import matplotlib.pyplot as plt

class Phasefield:
  def __init__(self, grid, problem):
    self.grid = grid
    self.problem = problem
    self.space = dglagrange(grid, order=1)
    self.d = TrialFunction(self.space)
    self.dd = TestFunction(self.space)
    self.dh = self.space.interpolate(1, name="dh")
    self.uh = 0


  def __compile(self, verbose=False):
    P = self.problem
    betad = P.g_c * P.l * P.beta0
    n = FacetNormal(self.space)
    h = avg( CellVolume(self.space) ) / FacetArea(self.space)
    d, dd = self.d, self.dd

    Ad  = -P.g_c / P.l * (1 - d) * dd * dx
    Ad += inner(P.g_c * P.l * grad(d), grad(dd)) * dx

    if self.uh != 0:
      Ad += P.dA(d) * P.Psi_p(self.uh) * dd * dx

    Ad += betad / h * inner(jump(d), jump(dd)) * dS
    Ad -= dot(dot(avg(P.g_c * P.l * grad(d)), n('+')), jump(dd)) * dS

    self.scheme = galerkin([Ad == 0], solver=("suitesparse", "umfpack"),
        parameters={"newton.verbose": verbose}
    )


  def solve(self, verbose=False):
    if not hasattr(self, 'scheme'):
      self.__compile(verbose=verbose)

    start = time()

    self.scheme.solve(self.dh)

    if verbose:
      print("Took {:.2f}s".format(time()-start))


  def plot(self, step, dir='out'):
    path = self.__getPath(dir, 'plot', 'png')

    plt.clf()
    plotPointData(self.dh, figure=plt.gcf(), gridLines=None, clim=[0,1],
      xlim=[0, self.problem.domainSize], ylim=[0, self.problem.domainSize])
    plt.savefig(path+'phasefield-'+str(step)+'.png', dpi=500)


  def __getPath(self, dir, prefix, type):
    name = self.problem.name
    directory = dir + '/' + name + '/' + type
    os.makedirs(directory, exist_ok=True)
    return directory + '/' + prefix + ('-' if prefix != '' else '')
