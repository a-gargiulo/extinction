import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("./data/extinction.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Access the lists by key
p, T_ext, a_ext, n = loaded_data

print(T_ext)
# Create the plot
fig, ax1 = plt.subplots()

# Plot the first dataset on the first y-axis (ax1)
ax1.semilogx(p, T_ext, 'b-')
ax1.set_xlabel('p')
ax1.set_ylabel('T_max', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()


a_plot = [x["mean"] for x in a_ext]

# Plot the second dataset on the second y-axis (ax2)
ax2.semilogx(p, a_plot, 'r-')
ax2.set_ylabel('a_max', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Set the second y-axis to logarithmic scale
ax2.set_yscale('log')

# Show the plot
plt.show()
