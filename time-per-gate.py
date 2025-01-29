import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = True
plt_size = 35


def Average(lst): 
    return sum(lst) / len(lst) 

name = 'dnn_n20'
smooth_n = 10
# Open and read data from the three files
file1 = open("log/results/time/"+name+"_FlatDD.txt", "r")
file2 = open("log/results/time/"+name+"_DDSIM.txt", "r")
file3 = open("log/results/time/"+name+"_QPP.txt", "r")

data2 = [float(line.strip()) for line in file2.readlines()]
data1 = []
for line in file1.readlines():
    if "Switch Overhead:" not in line:
        data1.append(float(line.strip()))
data3 = [float(line.strip()) for line in file3.readlines()]

# Close the files
file1.close()
file2.close()
file3.close()
# plt.figure(figsize=(4,6))
fig, ax = plt.subplots(1, 1, figsize=(8,8))
# ax.figure(figsize=(4,6))
smoothen1 = [Average(data1[i:min(i+smooth_n, len(data1))]) for i in range(0,len(data1),smooth_n)]
smoothen2 = [Average(data2[i:min(i+smooth_n, len(data2))]) for i in range(0,len(data2),smooth_n)]
smoothen3 = [Average(data3[i:min(i+smooth_n, len(data3))]) for i in range(0,len(data3),smooth_n)]

g = np.linspace(0,len(smoothen1), len(smoothen1)) * smooth_n
g1 = np.linspace(0,len(smoothen1)-1, len(smoothen1)-1) * smooth_n

# Plot the data
plt.plot(g, smoothen1, label=r"\texttt{FlatDD}", color = (57/255, 137/255, 201/255))
plt.plot(g, smoothen2, label=r"\texttt{DDSIM}", color = (211/255, 88/255, 74/255))
plt.plot(g, smoothen3, label=r"\texttt{Quantum++}", color='g')
ax.set_yscale('log', base=10)
# Add labels and legend
ax.set_xlabel(r"Gates", size = plt_size)
ax.set_ylabel(r"Runtime per gate", size = plt_size)
plt.yticks(size=plt_size)
plt.xticks(rotation=50, size=plt_size)
plt.subplots_adjust(left=0.19, bottom=0.19, right = 0.99, top = 0.99)
# plt.legend(prop = { "size": plt_size })
# bbox_to_anchor=(0.3, 0.4)
ax.legend(loc='upper left', bbox_to_anchor=(-0.04, 1.05), prop = { "size": plt_size }, frameon=False)
# Show the plot
plt.savefig("time-per-gate-"+name+".pdf")
plt.clf()

name = 'supremacy_n20'
smooth_n = 10
# Open and read data from the three files
file1 = open("log/results/time/"+name+"_FlatDD.txt", "r")
file2 = open("log/results/time/"+name+"_DDSIM.txt", "r")
file3 = open("log/results/time/"+name+"_QPP.txt", "r")

data2 = [float(line.strip()) for line in file2.readlines()]
data1 = []
for line in file1.readlines():
    if "Switch Overhead:" not in line:
        data1.append(float(line.strip()))
data3 = [float(line.strip()) for line in file3.readlines()]

# Close the files
file1.close()
file2.close()
file3.close()
# plt.figure(figsize=(4,6))
fig, ax = plt.subplots(1, 1, figsize=(8,8))
# ax.figure(figsize=(4,6))
smoothen1 = [Average(data1[i:min(i+smooth_n, len(data1))]) for i in range(0,len(data1),smooth_n)]
smoothen2 = [Average(data2[i:min(i+smooth_n, len(data2))]) for i in range(0,len(data2),smooth_n)]
smoothen3 = [Average(data3[i:min(i+smooth_n, len(data3))]) for i in range(0,len(data3),smooth_n)]

g = np.linspace(0,len(smoothen1), len(smoothen1)) * smooth_n
g1 = np.linspace(0,len(smoothen1)-1, len(smoothen1)-1) * smooth_n

# Plot the data
plt.plot(g, smoothen1, label=r"\texttt{FlatDD}", color = (57/255, 137/255, 201/255))
plt.plot(g, smoothen2, label=r"\texttt{DDSIM}", color = (211/255, 88/255, 74/255))
plt.plot(g, smoothen3, label=r"\texttt{Quantum++}", color='g')
ax.set_yscale('log', base=10)
# Add labels and legend
ax.set_xlabel(r"Gates", size = plt_size)
ax.set_ylabel(r"Runtime per gate", size = plt_size)
plt.yticks(size=plt_size)
plt.xticks(rotation=50, size=plt_size)
plt.subplots_adjust(left=0.19, bottom=0.19, right = 0.99, top = 0.99)
# plt.legend(prop = { "size": plt_size })
# bbox_to_anchor=(0.3, 0.4)
ax.legend(loc='upper left', bbox_to_anchor=(-0.04, 1.05), prop = { "size": plt_size }, frameon=False)
# Show the plot
plt.savefig("time-per-gate-"+name+".pdf")