import numpy as np
from time import time
from dune.fem import adapt
import io
from dune.generator import algorithm

# Add interface helper method
def addInterface(dh, interfacefunctions=None, threshold=0.01, verbose=False):
    code = \
"""
#include <iostream>
#include <algorithm>

template <class Grid, class GridView, class D>
bool addInterface(Grid& grid, GridView& gridView, const D& dh, const double threshold, const bool verbose)
{
  bool added = false;
  const auto& igridView = grid.interfaceGrid().leafGridView();

  typename D::LocalFunctionType localFunction( dh );
  typename D::RangeType dIn, dOut;

  auto criterion = [&] (const auto& i)
  {
    const auto& igeo = i.geometry();

    const auto& geo = i.inside().geometry();
    localFunction.bind(i.inside());
    localFunction.evaluate(geo.local(igeo.center()), dIn);

    const auto& ogeo = i.outside().geometry();
    localFunction.bind(i.outside());
    localFunction.evaluate(ogeo.local(igeo.center()), dOut);

    if (0.5 * (dIn + dOut) < threshold)
      return true;
    else
      return false;
  };

  static constexpr int dim = Grid::dimension;
  using Vertex = typename Grid::template Codim<dim>::Entity;

  for (const auto& e : elements(gridView))
  {
    for (const auto& i : intersections(gridView, e))
    {
      if (!grid.isInterface(i) and !i.boundary() and criterion(i))
      {
        if (verbose)
          std::cout << "Add interface at (" << i.geometry().center() << ")" << std::endl;
        grid.addInterface(i);
        added = true;
      }
    }
  }
  return added;
}
"""
    if interfacefunctions is not None:
      ph_g, ph_gOld = interfacefunctions

      p_gAvg = np.mean(ph_g.as_numpy)
      p_gMin = np.min(ph_g.as_numpy)
      p_gMax = np.max(ph_g.as_numpy)

    added = algorithm.run('addInterface', io.StringIO(code), dh.grid.hierarchicalGrid, dh.grid, dh, threshold, verbose)

    if added:
      if interfacefunctions is not None:
        # make sure to call adapt on interface
        adapt([ph_g, ph_gOld])

        # initialize new interface elements with average value, care also about NaNs
        def init(ph_g):
          v = ph_g.as_numpy
          v[v < p_gMin] = p_gAvg
          v[v > p_gMax] = p_gAvg
          v[np.isnan(v)] = p_gAvg

        init(ph_g)
        init(ph_gOld)

    return added
