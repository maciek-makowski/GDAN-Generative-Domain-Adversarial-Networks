import numpy as np
import matplotlib.pyplot as plt

npz_file = np.load('best_solution_list.npz')

#print("Keys in the NPZ file:", npz_file.files)

data_for_plotting = {i: [] for i in range(11)}

# Access and print the contents of each array
for key in npz_file.files:
    #print(f"{key}: {npz_file[key]}")
    for i in range(11):
        data_for_plotting[i].append(npz_file[key][i])


#print(data_for_plotting[0])

fig, axs = plt.subplots(4,3, figsize = (16,9))
temp = 0
for i in range(4):
    for j in range(3):
        if i == 3 and j ==2:            
            break 

        axs[i,j].hist(data_for_plotting[temp], bins=20, color='skyblue', edgecolor='black')
        axs[i,j].set_xlabel(f"Feature {temp}")
        axs[i, j].grid(True)
        temp = temp + 1
    

fig.suptitle('Distribution of weights for each feature', fontsize=16)
#plt.subplots_adjust(bottom=0.1)

plt.tight_layout()
plt.show()


npz_file.close()