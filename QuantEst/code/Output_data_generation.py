import numpy as np
import os

# Check if folder exists, if not, then create new empty folders
# under the assumption that the main folder exists, create new folders in the main folder
# for the folders needs more branches (sub_fd_lst), the branches are the subsub_fd_lst 
def initialize_folders(main_fd, fd_lst, sub_fd_lst, subsub_fd_lst):
    
    # check list
    input_lst = [fd_lst, sub_fd_lst, subsub_fd_lst]
    for f in (input_lst):
        if not isinstance(f, list):
            raise Exception ("input {} should be a list!".format(f))
    # print (fd_lst, sub_fd_lst, subsub_fd_lst)

    for fd in fd_lst:
        if not os.path.exists(main_fd+fd):
            os.makedirs(main_fd+fd)

    for ss_fd in subsub_fd_lst:
        for s_fd in sub_fd_lst:
            fd_name = main_fd+s_fd+ss_fd
            if not os.path.exists(fd_name):
                os.makedirs(fd_name)


def get_settings(distro_lst, datasize_lst, stepsize_lst):
    len_lst = [len(distro_lst), len(datasize_lst), len(stepsize_lst)]
    if len_lst.count(1) != len(len_lst)-1:             
        raise Exception("Setting inputs are wrong!")
    
    N_settings = max((len_lst))
    setting_lst = []
    for lst in [distro_lst, datasize_lst, stepsize_lst]:
        if len(lst)==1: 
            lst = lst*N_settings
        setting_lst.append(lst)
        
    changed_setting = None
    change_lst = ['distro', 'data_size', 'step_size']
    for idx, l in enumerate(len_lst):
        if l>1: changed_setting = change_lst[idx]
    return np.asarray(setting_lst).T, changed_setting



def get_file_name(changed_setting, distro, datasize, stepsize, s_test):
#     print(changed_setting)
    if s_test:
        return ('Shuffle', 'shuffle')
    setting_dict = {
        'distro': ('Distribution', distro),
        'data_size': ('Data size', datasize),
        'step_size': ('Step size', stepsize),
    }
    if not setting_dict.get(changed_setting): 
        raise Exception ('Cannot get file name')
    return setting_dict.get(changed_setting)


def write_data(data, filename, setting=None):
    with open(filename, 'w') as outfile:
        if setting is not None: outfile.write('# Setting: {0}\n'.format(setting))
        if len(data.shape) == 1: data = data.reshape(-1, 1)
        outfile.write('# Array shape: {0}\n \n'.format(data.shape))

        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt='%-15.8g')
            outfile.write('\n')


def write_data_overview(category, setting, tau_lst, method_name, file_name):
    with open(file_name, 'w') as f:
        f.write("Tested on "+category+": "+str(setting)+"\n") 
        f.write(str(tau_lst)+'\n')
        f.write(method_name)
    
# write_data_overview('ca', 'se', tau_vals, "try.txt")
    
def save_data(foldername, file_name, tau_lst, data_dict, method_name):
    category, setting = file_name[0], file_name[1]    
    write_data_overview(category, setting, tau_lst, method_name, foldername+str(setting)+"_"+"overview.txt")

    for data_name in data_dict:
        print (foldername+str(setting)+"_"+data_name+'.txt')
        write_data(data_dict[data_name], foldername+str(setting)+"_"+data_name+'.txt')
        
    return

# save_data('', ('C', 'S'), tau_vals, {'dataset': np.reshape(np.random.uniform(0, 4, 100), (2, 5, 10))})