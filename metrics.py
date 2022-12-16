import re
import numpy as np
import matplotlib.pyplot as plt

acc = []
ppl = []

with open('log', 'r') as f:
    for line in f:
        if re.search(r"acc:", line):
            data = "".join(line.split(";")[1:3]).strip().split(" ")
            acc.append(float(data[1]))
            ppl.append(float(data[-1]))

x = np.array(range(0, len(acc)*50, 50)) / 10000

fig, ax1 = plt.subplots(figsize=(24, 8))
ax2 = ax1.twinx()


ax1.plot(x, acc, c='C0')
ax2.plot(x, ppl, c='C1')

ax1.set_xlabel("Model", fontsize=30)
ax1.set_ylabel("ACC", c='C0', fontsize=30)
ax2.set_ylabel("PPL", c='C1', fontsize=30)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)

plt.savefig("metrics.png")
