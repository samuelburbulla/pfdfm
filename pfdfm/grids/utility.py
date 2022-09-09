import gmsh
import numpy as np

def symmetrizeMesh(transformCoord):

    def getModelEntities():
        m = {}
        elemax = 0
        nodemax = 0
        entities = gmsh.model.getEntities()
        res = [[ i for i, j in entities ],
               [ j for i, j in entities ]]

        entmax = max(res[1])

        for e in entities:
            bnd = gmsh.model.getBoundary([e])
            nod = gmsh.model.mesh.getNodes(e[0], e[1])
            ele = gmsh.model.mesh.getElements(e[0], e[1])
            m[e] = (bnd, nod, ele)
            if len(nod[0]) != 0:
                if nodemax < max(nod[0]):
                    nodemax = max(nod[0])
            if ele[1]:
                if elemax < max(ele[1][0]):
                    elemax = max(ele[1][0])
        return m, entmax, nodemax, elemax


    def transform(m, offset_entity, offset_node, offset_element):
        for e in sorted(m):
            dimTags, nodes, elements = m[e]
            newDimTag = e[1] + offset_entity

            boundary = [b[1] + offset_entity for b in dimTags]
            gmsh.model.addDiscreteEntity(e[0], newDimTag, boundary)

            coord = []
            for i in range(0, len(nodes[1]), 3):
                x, y, z = nodes[1][i], nodes[1][i+1], nodes[1][i+2]
                xn, yn, zn = transformCoord(x, y, z)
                coord.append(xn)
                coord.append(yn)
                coord.append(zn)
            gmsh.model.mesh.addNodes(e[0], newDimTag, nodes[0] + offset_node, coord)

            gmsh.model.mesh.addElements(e[0], newDimTag, elements[0],
                                        [t + offset_element for t in elements[1]],
                                        [n + offset_node for n in elements[2]])

        gmsh.model.mesh.removeDuplicateNodes()


    m, entmax, nodemax, elemax = getModelEntities()
    transform(m, entmax, nodemax, elemax)



# Remove codim-1 elements from mesh file
def removeCodim1Elements(file):
    outfile = ""
    placeholder = "###"
    elementBlock = -1
    codim1elements = 0
    oldNumberOfElements = 0

    f = open(file, "r")
    for line in f:
      if elementBlock == 0:
        oldNumberOfElements = int(line)
        elementBlock = 1
        outfile += placeholder+"\n"
        continue

      if line == '$Elements\n':
        elementBlock = 0
      if line == '$EndElements\n':
        elementBlock = -1

      if elementBlock > 0:
        data = line.split(" ")
        if int(data[1]) == 1:
          codim1elements += 1
          continue

      outfile += line

    outfile = outfile.replace(placeholder, str(oldNumberOfElements-codim1elements))
    f = open(file, "w")
    f.write(outfile)
