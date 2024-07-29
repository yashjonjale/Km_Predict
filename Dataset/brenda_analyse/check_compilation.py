import glob
import os




pattern1 = os.getcwd() + "/ec_grps/*.csv"
pattern2 = os.getcwd() + "/seqs/*.csv"

files1 = glob.glob(pattern1)
files2 = glob.glob(pattern2)

print(f"Total files in ec_grps: {len(files1)}")
print(f"Total files in seqs: {len(files2)}")
    