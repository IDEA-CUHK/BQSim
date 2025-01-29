import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re

t = [1, 2, 4, 8, 16]
saved_cost = []
upbound = 15
up_idx = 1
with open("log/cache.log") as file:
    for line in file:
        if "Saved cost %:" in line:
            costs = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
            saved_cost.append(float(costs[0]))
            up_idx += 1
        if up_idx > upbound:
            break

t = [1, 2, 4, 8, 16]
sim_time = []
sim_acc = []
upbound = 30
up_idx = 1
with open("log/cache.log") as file:
    for line in file:
        if "simulation_time" in line:
            times = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
            sim_time.append(float(times[0]))
            up_idx += 1
        if up_idx > upbound:
            break
print(sim_time)
for i in range(15):
    sim_acc.append(100*(sim_time[i+15]-sim_time[i])/sim_time[i])

print(sim_acc)
print(saved_cost)
t3 = t * 3
with open('log/speedup-cache.csv', 'w') as f:
    f.write('threads,circuits\n') 
    for i in range(15):
        f.write(str(t3[i])+","+str(sim_acc[i])+"\n")
with open('log/cost-cache.csv', 'w') as f:
    f.write('threads,circuits\n') 
    for i in range(15):
        f.write(str(t3[i])+","+str(saved_cost[i])+"\n")

# # plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['text.usetex'] = True

plt_size = 20
df = pd.read_csv('log/speedup-cache.csv')
plt.figure(figsize=(6,6))
fig, ax = plt.subplots()
sns.lineplot(x="threads", y="circuits", data=df, color = (57/255, 137/255, 201/255))
plt.yticks(size=plt_size)
plt.xticks(size=plt_size)
vals = ax.get_yticks()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xscale('log', base=2)
plt.xticks(t, t, size=plt_size)
plt.subplots_adjust(left=0.22, bottom=0.17, right = 0.99, top = 0.94)
ax.set_xlabel(r"Threads (t)", size = plt_size)
ax.set_ylabel(r"Speed-up", size = plt_size)
plt.savefig("log/speedup-cache.pdf")

df = pd.read_csv('log/cost-cache.csv')
plt.figure(figsize=(6,6))
fig, ax = plt.subplots()
sns.lineplot(x="threads", y="circuits", data=df, color = (57/255, 137/255, 201/255))
plt.yticks(size=plt_size)
plt.xticks(size=plt_size)
vals = ax.get_yticks()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xscale('log', base=2)
plt.xticks(t, t, size=plt_size)
plt.subplots_adjust(left=0.22, bottom=0.17, right = 0.99, top = 0.94)
ax.set_xlabel(r"Threads (t)", size = plt_size)
ax.set_ylabel(r"Comp. Cost Reduction", size = plt_size)
plt.savefig("log/cost-cache.pdf")