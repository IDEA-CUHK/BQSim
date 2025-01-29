# import matplotlib.pyplot as plt
# import numpy as np

# x1 = np.array(["32", "64", "128", "256", "512"])
# y1 = np.array([7.31223825, 8.743636619, 9.570996979, 10.11734254, 10.32740243])

# plt.bar(x1,y1)
# plt.savefig("dnn_n17_scala.pdf", format="pdf")
# plt.clf()

# y2 = np.array([2.156808803, 2.320074349, 2.413497172, 2.474833232, 2.522749473])
# plt.bar(x1,y2)
# plt.savefig("vqe_n16_scala.pdf", format="pdf")
# plt.clf()

# y3 = np.array([4.288450377, 4.798679868, 5.013986014, 5.067727313, 5.090750045])
# plt.bar(x1,y3)
# plt.savefig("port_vqe_n16_scala.pdf", format="pdf")
# plt.clf()

# y4 = np.array([1.816829746, 2.03330373, 2.117227676, 2.154510785, 2.140096185])
# plt.bar(x1,y4)
# plt.savefig("tsp_n16_scala.pdf", format="pdf")
# plt.clf()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.family"] = "Times New Roman"
labels = np.array(["32", "64", "128", "256", "512"])
y1 = np.array([7.31223825, 8.743636619, 9.570996979, 10.11734254, 10.32740243])
y2 = np.array([2.156808803, 2.320074349, 2.413497172, 2.474833232, 2.522749473])
y3 = np.array([4.288450377, 4.798679868, 5.013986014, 5.067727313, 5.090750045])
y4 = np.array([1.816829746, 2.03330373, 2.117227676, 2.154510785, 2.140096185])

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots(figsize=(5, 5))
rects1 = ax.bar(x , y1, width, label=r'DNN $n=17$')
# rects2 = ax.bar(x - width/2, y2, width, label=r'VQE $n=16$')
# rects3 = ax.bar(x + width/2, y3, width, label=r'Port. opt. w/ VQE $n=16$')
# rects4 = ax.bar(x + 3*width/2, y4, width, label=r'TSP $n=16$')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Batch size')
ax.set_ylabel('Speed-up')
ax.set_xticks(x)
ax.set_xticklabels(labels)
# ax.legend()
plt.savefig("dnn_n17_scala.pdf", format="pdf")