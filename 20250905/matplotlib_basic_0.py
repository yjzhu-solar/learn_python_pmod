import numpy as np 
import matplotlib.pyplot as plt



a = np.array([1,2,3])
b = np.array([2,3,4])

# c = [1,2,3]
# print(c*3)

# plt.plot(a,b)
# plt.plot(a-1,b)

# one axis
fig, ax = plt.subplots()
ax.plot(a,b, color="red")

print(ax.get_xlim())
#print(dir(ax))
plt.show()
