import matplotlib.pyplot as plt
import numpy as np

cpu_time =[]
gpu_time = []
node = []
edge = []
qubit = []

filename = 'data'

with open(filename+'.txt') as f:
    for line in f: # read rest of lines
        words = line.split()
        numbers = [int(word) for word in words if word.replace('.', '', 1).isdigit()] 
        if len(numbers) == 5:
            # array = [int(x) for x in line.split()]
            # if (array[4] == 20):
            gpu_time.append(numbers[0])
            cpu_time.append(numbers[1])
            node.append(numbers[2])
            edge.append(numbers[3])
            qubit.append(numbers[4])

gpu_cpu = np.array(gpu_time)-np.array(cpu_time)
cpu_gpu = -np.array(gpu_time)+np.array(cpu_time)

gpu_less_time = []
gpu_less_edge = []

cpu_less_time = []
cpu_less_edge = []

for i in range(len(gpu_cpu)):
    if gpu_cpu[i] > 0:
        cpu_less_time.append(cpu_time[i])
        cpu_less_edge.append(edge[i])
    if cpu_gpu[i] > 0:
        gpu_less_time.append(gpu_time[i])
        gpu_less_edge.append(edge[i])

plt.scatter(cpu_less_edge, cpu_less_time, alpha=0.4)
plt.scatter(gpu_less_edge, gpu_less_time, alpha=0.4)
# plt.ylim(bottom=1.5, top=10**8)
# plt.scatter(qubit, cpu_time, alpha=0.4)
plt.yscale("log")
plt.xscale("log")
plt.legend(["CPU takes less time", "GPU takes less time"])
plt.title("DD-to-ELL conversion (GPU vs. CPU)")
plt.xlabel("Num. of edges")
plt.ylabel("Gate conversion time (ms)")


print("CPU total time: "+str(sum(cpu_time)))
print("GPU total time: "+str(sum(gpu_time)))

min_sum = 0
for i in range(len(cpu_time)):
    min_sum += gpu_time[i] if (cpu_time[i] > gpu_time[i])  else cpu_time[i]
print("Mixed min total time: "+str(min_sum))
plt.savefig(filename+".pdf", format="pdf")
