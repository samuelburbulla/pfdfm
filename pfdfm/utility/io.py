import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

def writeToFilesAndPlot(map, dir, plot=True):
  for entry in map:
    f = open(dir+'/'+entry.lower()+".csv", "w")
    f.write("t, "+entry+"\n")
    for line in map[entry]:
      f.write(", ".join([str(l) for l in line])+"\n")
    f.close()

  if plot:
    for entry in map:
      data = map[entry]
      x = [d[0] for d in data]
      y = [d[1] for d in data]
      plt.clf()
      plt.grid()
      plt.gcf().subplots_adjust(left=0.15)
      plt.plot(x, y, label=entry)
      plt.legend(loc='best')
      plt.savefig(dir+'/'+entry.lower()+".png", dpi=500)
