import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

dir = 'out'

plots = {
  'length':
  {
    '$l = 0.08$': ('horizontal-0.04/length.csv', 'darkgrey'),
    '$l = 0.04$': ('horizontal-0.02/length.csv', 'grey'),
    '$l = 0.02$': ('horizontal-0.01/length.csv', 'black')
  },
  'energy':
  {
    '$W(\mathbf{u}, p, d, p_\Gamma)$': 'horizontal-0.01/energy.csv',
    '$Q(t)$': ('horizontal-0.01/source.csv', 'black')
  },
  'pressure':
  {
    'Pressure': 'horizontal-0.01/pressure.csv',
    'Reference': ('reference/pressure.csv', 'k--')
  },
  'aperture':
  {
    'Aperture': 'horizontal-0.01/aperture.csv',
    'Reference': ('reference/aperture.csv', 'k--')
  },
  'iterations1':
  {
    'Iterations': 'horizontal-0.01/iterations.csv',
  },
  'iterations2':
  {
    'Iterations': 'joining/iterations.csv',
  },
}

for title in plots:
  plt.clf()
  files = plots[title]

  for label in files:
    file = files[label]
    args = []
    if isinstance(file, tuple):
      file, *args = file
    xs, ys = [], []
    f = open(dir+'/'+file, "r")
    l = 0
    for line in f:
      l += 1
      if l == 1:
        continue
      x, y = line.split(",")
      xs += [float(x)]
      ys += [float(y)]
    f.close()

    plt.plot(xs, ys, label=label, *args)

  plt.grid()
  plt.gcf().subplots_adjust(left=0.15)
  plt.legend(loc='best')
  plt.savefig(dir+'/'+title.lower()+".png", dpi=500)
