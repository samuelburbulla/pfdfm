from dune.grid import reader
from dune.mmesh import mmesh, interfaceIndicator, normals
from dune.fem.view import adaptiveLeafGridView
from pfdfm.problems.common import Common
from pfdfm.models.coupled import Coupled
from pfdfm.models.poroelastic import Poroelastic
from pfdfm.utility.interface import addInterface
from pfdfm.utility.io import writeToFilesAndPlot
from dune.fem.function import integrate
from dune.fem import adapt
import numpy as np
from time import time

verbose = True
plot = True

start = time()
output = {
  'Energy': [],
  'Source': [],
  'Pressure': [],
  'Length': [],
  'Aperture': [],
  'Iterations': []
}

problem = Common()
mesh = mmesh((reader.gmsh, problem.mshfile), problem.dim)
problem.meshfile.postprocess(mesh)
grid = adaptiveLeafGridView(mesh)
hgrid = grid.hierarchicalGrid
igrid = adaptiveLeafGridView(hgrid.interfaceGrid)

coupled = Coupled(Poroelastic, grid, problem)
poroelastic = coupled.elasticity
print("DOFs (sol + ph_g + dh):", poroelastic.sol.size, "+", poroelastic.ph_g.size, "+", coupled.phasefield.dh.size)

poroelastic.write(step=0)
if plot:
  poroelastic.plot(step=0)
  coupled.phasefield.plot(step=0)

t = 0
step = 0

output['Energy'] += [(t, problem.W(poroelastic, coupled.phasefield.dh))]
output['Source'] += [(t, problem.Q(poroelastic))]

while t + problem.dt < problem.T + 1e-14:
  step += 1
  t += problem.dt
  print(f"--- Step {step}: t = {t:.4g} ---")

  iterations = 0
  repeat = True
  while repeat:
    iterations += coupled.solve(verbose)

    output['Energy'] += [(t, problem.W(poroelastic, coupled.phasefield.dh))]

    repeat = addInterface(
      coupled.phasefield.dh,
      [poroelastic.ph_g, poroelastic.ph_gOld],
      threshold=problem.threshold,
      verbose=True
    )

    poroelastic.write(step)
    if plot:
      poroelastic.plot(step)

    if repeat:
      coupled.phasefield.dh.interpolate(1)

      # update interfaceIndicator and normals (needed in dune-mmesh<=1.3.2)
      adapt([igrid.hierarchicalGrid.one, poroelastic.normal])
      igrid.hierarchicalGrid.one.interpolate(1)
      poroelastic.normal.assign(normals(igrid))

  # Output of iterations, free isothermal energy, source, maximum fracture pressure, fracture length and maximum aperture
  output['Iterations'] += [(t, iterations)]
  output['Energy'] += [(t, problem.W(poroelastic, coupled.phasefield.dh))]
  output['Source'] += [(t, problem.Q(poroelastic, proceed=True))]
  output['Pressure'] += [(t, np.max(poroelastic.ph_g.as_numpy))]
  output['Length'] += [(t, integrate(igrid, 1, order=0))]
  output['Aperture']  += [(t, np.max(poroelastic.ispace.interpolate(poroelastic.w, name="w0").as_numpy))]
  writeToFilesAndPlot(output, 'out/'+problem.name, plot=plot)

  poroelastic.proceed()

print("Overall runtime is {:.2f}s.".format(time()-start))
