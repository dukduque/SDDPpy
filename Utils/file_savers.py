'''
Created on Jul 2, 2018

@author: dduque

Various methods to save file 
'''

import pickle 


def write_object_results(full_path, result_to_save):
    #Pickling
    try:
        file = open(full_path, 'wb')
        pickle.dump(result_to_save,file)
    except:
        print('Error while saving file %s' %full_path)