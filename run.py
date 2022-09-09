import subprocess

# Problem 1 #
for hf in [0.04, 0.02, 0.01]:
  cmd = 'python poroelasticity.py --hf '+str(hf)
  print(cmd)
  subprocess.run(cmd.split(" "))

# Problem 2 #
cmd = 'python poroelasticity.py --problem 2'
print(cmd)
subprocess.run(cmd.split(" "))

# Plot #
cmd = 'python plot.py'
print(cmd)
subprocess.run(cmd.split(" "))
