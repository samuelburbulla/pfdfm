import gmsh
from pfdfm.grids.utility import symmetrizeMesh, removeCodim1Elements

def create(h, hf, l=4):
    name = "tip"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.add(name)
    geo = gmsh.model.geo

    p1 = geo.addPoint(0, 0, 0, h)
    p2 = geo.addPoint(l, 0, 0, h)
    p3 = geo.addPoint(l, l/2, 0, h)
    p4 = geo.addPoint(0, l/2, 0, h)

    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)

    geo.addCurveLoop([l1, l2, l3, l4], 1)
    geo.addPlaneSurface([1], 0)

    geo.synchronize()

    def mesh_size(entity_dim, entity_tag, x, y, z, lc=0):
        return hf + (h - hf) * abs(y - l/2.) / (l/2.)

    gmsh.model.mesh.setSizeCallback(mesh_size)

    gmsh.model.mesh.generate(dim=2)

    def transformCoord(x, y, z):
      return x, 4 - y, z

    symmetrizeMesh(transformCoord)

    file = "/tmp/"+name+".msh"
    gmsh.write(file)

    removeCodim1Elements(file)

    gmsh.finalize()
    return file


def postprocess(grid, l=4):
    hgrid = grid.hierarchicalGrid
    for e in grid.elements:
      for i in grid.intersections(e):
        x = i.geometry.center
        if x[0] > l/2-l/20. and x[0] < l/2+l/20. and abs(x[1] - l/2) < 1e-6:
          hgrid.addInterface(i)
