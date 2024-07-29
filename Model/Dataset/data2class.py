import numpy as np

file = "/home/yashjonjale/Documents/intern_proj/Model/Dataset/final_data.npz"

#extract cmd parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, help="Threshold value for classification", required=True)
args = parser.parse_args()
th = args.threshold



data = np.load(file)
arr = data['arr_0']
print("_"*50)
print(f"Adapting the Data for Classification with threshold {th}")
print(f"The shape of the data is {arr.shape}")

indices = np.where(arr[:,2] <th)
indices_ = np.where(arr[:,2] >=th)
print()
print("Before - ")
print(arr[:5,[0,1,2,-1]])

print()
arr[indices,2] = 1
arr[indices_,2] = 0

print()
print("After - ")
#print the first five rows of arr
print(arr[:5,[0,1,2,-1]])
np.savez("/home/yashjonjale/Documents/intern_proj/Model/Dataset/data_binary.npz", arr)
#save arr as npz


print()
print("Done!")

print(f"Yes points - {len(indices[0])} \n No points - {len(indices_[0])}")
print("_"*50)



