from time import time
from pfdfm.models.phasefield import Phasefield
from dune.fem.function import integrate

class Coupled:
  def __init__(self, Model, grid, problem, igrid=None):
    self.grid = grid

    self.elasticity = Model(grid, problem, igrid=igrid, order=2)
    self.phasefield = Phasefield(grid, problem)

    self.elasticity.dh = self.phasefield.dh
    self.phasefield.uh = self.elasticity.uh

    self.dhTmp = self.phasefield.dh.copy()


  def solve(self, verbose=False, step=None):
    if step is not None:
        self.elasticity.write(step=step, prefix="filtered")

    start = time()
    j = 0
    res = 1
    while res > 1e-3 or j < 2:
      self.dhTmp.assign(self.phasefield.dh)
      self.elasticity.solve(verbose)
      self.phasefield.solve(verbose)

      if step is not None:
        self.elasticity.write(step=step, prefix="filtered")

      res = integrate(self.grid, (self.phasefield.dh - self.dhTmp)**2, order=2)**.5

      if verbose:
        print(" j: {:d} res = {:.4g}".format(j, res))

      j += 1

    if verbose:
      print("Coupled took {:.2f}s".format(time()-start))

    return j
