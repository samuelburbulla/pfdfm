import gmsh
from pfdfm.grids.utility import symmetrizeMesh, removeCodim1Elements

def create(h, hf, l=4):
    name = "joining"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.add(name)
    geo = gmsh.model.geo

    p1 = geo.addPoint(0, 0, 0, h)
    p2 = geo.addPoint(l, 0, 0, hf)
    p3 = geo.addPoint(l, l/2, 0, hf)
    p4 = geo.addPoint(l/2+3*l/20., l/2, 0, hf)
    p5 = geo.addPoint(l/2-l/20., l/2, 0, hf)
    p6 = geo.addPoint(0, l/2, 0, hf)
    p8 = geo.addPoint(l/2+3*l/20., l/2-l/20., 0, hf)

    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p5)
    l5 = geo.addLine(p5, p6)
    l6 = geo.addLine(p6, p1)
    lf = geo.addLine(p4, p8)

    cl = geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    rect = geo.addPlaneSurface([cl])

    geo.synchronize()
    gmsh.model.mesh.embed(1, [lf], 2, rect)
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
        if x[0] > l/2-l/20. and x[0] < l/2+2*l/20. and abs(x[1] - l/2) < 1e-6:
          hgrid.addInterface(i)

        if abs(x[0] - (l/2+3*l/20.)) < 1e-6 and x[1] > l/2-l/20. and x[1] < l/2+l/20.:
          hgrid.addInterface(i)
