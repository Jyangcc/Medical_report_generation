import os
import numpy as np


# Set the absolute path to the main directory
# dir_path = os.path.expanduser('/Volumes/YCC_MI/Kang_Ning_General_Hospital')

dir_path = 'Kang_Ning_General_Hospital/'

# List of child directories to process
child_dir = ['20230721_1st', '20230728_2nd', '20230804_3rd']

file_name_list = ['I0000000.dcm', 'I0000001.dcm','I0000002.dcm', 'I0000003.dcm']

not_full_files = []

for child in child_dir:
    # See every directory in child directory have how many files
    full_child_path = os.path.join(dir_path, child)
    files = os.listdir(full_child_path)
    print(f"Directory: {full_child_path} has {len(files)} files.") 
    
    
    file_cnt = []
    
    # get directory of each file
    for file in files:
        full_file_path = os.path.join(full_child_path, file)

        if os.path.isdir(full_file_path):
            files = os.listdir(full_file_path)
            # if all file in file_name_list are in files
            if all(item in files for item in file_name_list):
                # print(files)
                file_cnt.append(len(files))
            else:
                not_full_files.append(full_file_path)
                # print(f"Not full files in {full_file_path}")
                

    # How many = 4 files
    unique, counts = np.unique(file_cnt, return_counts=True)
    file_distribution = dict(zip(unique, counts))
    print(f"File distribution in {full_child_path}:\n {file_distribution}\n")# For each child directory, count the number of files in each subdirectory

# save the not full files path to txt
with open('docs/not_full_files.txt', 'w') as f:
    for item in not_full_files:
        f.write("%s\n" % item)