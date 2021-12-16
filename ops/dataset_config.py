# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

# 일단 root 는 다 절대경로로 ㄲ


######################
# ucf(7) -> hmdb(14) #
######################
def return_ucf101_uh(modality):
    filename_categories = 8
    if modality == 'RGB':
        root_data = '/data/dataset/ucf101/rawframes'
        filename_imglist_train = '/data/gyeongho/framework/tsm_bp/dataset/ufc-hmdb/ucf101_train_uh.txt'
        filename_imglist_val = '/data/gyeongho/framework/tsm_bp/dataset/ufc-hmdb/ucf101_valid_uh.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51_uh(modality):
    filename_categories = 8
    if modality == 'RGB':
        root_data = '/data/dataset/hmdb51/rawframes'
        filename_imglist_train = '/data/gyeongho/framework/tsm_bp/dataset/ufc-hmdb/hmdb_train_uh.txt'
        filename_imglist_val = '/data/gyeongho/framework/tsm_bp/dataset/ufc-hmdb/hmdb_valid_uh.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



######################
# hmdb(14) -> ucf(7) #
######################

def return_ucf101_hu(modality):
    filename_categories = 8
    if modality == 'RGB':
        root_data = '/data/dataset/ucf101/rawframes'
        filename_imglist_train = '/data/gyeongho/framework/tsm_bp/dataset/hmdb-ufc/ucf101_train_hu.txt'
        filename_imglist_val = '/data/gyeongho/framework/tsm_bp/dataset/hmdb-ufc/ucf101_valid_hu.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51_hu(modality):
    filename_categories = 8
    if modality == 'RGB':
        root_data = '/data/dataset/hmdb51/rawframes'
        filename_imglist_train = '/data/gyeongho/framework/tsm_bp/dataset/hmdb-ufc/hmdb_train_hu.txt'
        filename_imglist_val = '/data/gyeongho/framework/tsm_bp/dataset/hmdb-ufc/hmdb_valid_hu.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


# to do



def return_dataset(dataset, modality):
    dict_single = {
                   'ucf101-uh': return_ucf101_uh, 'hmdb51-uh': return_hmdb51_uh,
                   'ucf101-hu': return_ucf101_hu, 'hmdb51-hu': return_hmdb51_hu,
                     }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(file_imglist_train)
    file_imglist_val = os.path.join(file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
