import cv2
import numpy as np
import os
from torchvision.transforms import Compose, CenterCrop
import torch



def crop_transform(picture_size):
    return Compose([
        CenterCrop(picture_size)])


def load_data(file_dir, asc_dir, id=0, picture_size=88, setting='SOC', is_train=True):
    if setting == 'SOC':
        label_name = {'BMP2SN_9563': 0, 'BTR70SN_C71': 1, 'T72SN132': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7,
                      'ZIL131': 8, 'ZSU_23_4': 9}
    elif setting == 'EOC-2':
        label_name = {'2S1': 0, 'BRDM_2': 1, 'ZSU_23_4': 2, 'A64': 3}
    elif setting == 'EOC-1':
        label_name = {'BRDM_2': 0, 'ZSU_23_4': 1, '2S1': 2}
    elif setting == 'EOC-3':
        label_name = {'T72SN132': 0, 'BMP2SN_9563': 1,
                      'BRDM_2': 2, 'BTR70SN_C71': 3}
    elif setting == 'FUSAR':
        label_name = {'Cargo': 0, 'Tanker': 1, 'Bulkcarrier': 2}

    path_list = []
    asc_path_list = []
    jpeg_list = []
    asc_list = []
    label_list = []

    for files in os.listdir(file_dir):

        file_names = os.listdir(os.path.join(file_dir, files))
        for file_name in file_names:
            path_list.append(os.path.join(file_dir, files, file_name))

    if setting == 'FUSAR':
        for root, dirs, files in os.walk(asc_dir):
            files = sorted(files)
            for file in files:
                if len(file.split('_ASC.jpg')) == 2:
                    asc_path_list.append(os.path.join(root, file))
    else:
        for root, dirs, files in os.walk(asc_dir):
            files = sorted(files)
            for file in files:
                if len(file.split('_')) == 2:
                    asc_path_list.append(os.path.join(root, file))
    transform = crop_transform(picture_size)

    for jpeg_path in path_list:
        jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)

        pic = transform(torch.from_numpy(jpeg))

        jpeg_list.append(np.array(pic.div(pic.max())))
        name_list = jpeg_path.split('\\')
        for i in range(len(name_list)):
            if name_list[i] in label_name.keys():
                label_index = i
        label_list.append(label_name[name_list[label_index]])

    rate = 1.0
    assert len(asc_path_list) == len(
        path_list) == len(label_list) == len(jpeg_list)
    len_sample = round(len(asc_path_list)*rate)
    asc_path_list = asc_path_list[:len_sample]
    jpeg_list = jpeg_list[:len_sample]
    path_list = path_list[:len_sample]
    label_list = label_list[:len_sample]

    for asc_path, jpeg_path in zip(asc_path_list, path_list):
        asc_name = asc_path.split('\\')[-1].split('_ASC.jpg')[0]
        if setting == 'FUSAR':
            jpg_name = jpeg_path.split('\\')[-1].split('.png')[0]
        else:
            jpg_name = jpeg_path.split('\\')[-1].split('.JPG')[0]
        assert asc_name == jpg_name, "asc数据与图像数据名称不一致, 检查数据集"
        asc_jpeg = cv2.imread(asc_path, cv2.IMREAD_GRAYSCALE)

        pic = crop_transform(picture_size)(torch.from_numpy(asc_jpeg))

        asc_list.append(np.array(pic.div(pic.max())))

    jpeg_data = np.array(jpeg_list)
    asc_data = np.array(asc_list)
    jpeg_data = np.expand_dims(jpeg_data, axis=1)
    asc_data = np.expand_dims(asc_data, axis=1)
    # mask = sailency(data=data, id=id)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(jpeg_data).type(torch.FloatTensor),
                                              torch.from_numpy(asc_data).type(
                                                  torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))
    return data_set, label_name


def load_test(file_dir, asc_dir, id=0, picture_size=128, setting='SOC'):
    if setting == 'SOC':
        label_name = {'BMP2SN_9563': 0, 'BTR70SN_C71': 1, 'T72SN132': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7,
                      'ZIL131': 8, 'ZSU_23_4': 9}
    elif setting == 'EOC-2':
        label_name = {'2S1': 0, 'BRDM_2': 1, 'ZSU_23_4': 2, 'A64': 3}
    elif setting == 'EOC-1':
        label_name = {'BRDM_2': 0, 'ZSU_23_4': 1, '2S1': 2}
    elif setting == 'EOC-3':
        label_name = {'T72-812': 0, 'T72-A04': 0, 'T72-A05': 0,
                      'T72-A07': 0, 'T72-A10': 0, 'BMP2SN_9566': 1, 'BMP2SN_C21': 1, }
    elif setting == 'FUSAR':
        label_name = {'Cargo': 0, 'Tanker': 1, 'Bulkcarrier': 2}

    path_list = []
    asc_path_list = []
    jpeg_list = []
    asc_list = []
    label_list = []
    # load jpg list
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        if setting == 'EOC-1':
            # files = files[-303:] # EOC-1 分角度测试结果 ，此为45度情况
            files = files[:]
        for file in files:
            if setting == 'FUSAR':
                if os.path.splitext(file)[1] == '.png':
                    path_list.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[1] == '.JPG':
                    path_list.append(os.path.join(root, file))

    # load asc list
    for root, dirs, files in os.walk(asc_dir):
        file_list = []
        files = sorted(files)
        for file in files:
            if len(file.split('_')) == 2:
                file_list.append(file)
        if setting == 'EOC-1':
            # files = file_list[-303:]
            files = file_list
        if setting == 'FUSAR':
            for file in files:
                if len(file.split('_ASC.jpg')) == 2:
                    asc_path_list.append(os.path.join(root, file))
        else:
            for file in files:
                if len(file.split('_')) == 2:
                    asc_path_list.append(os.path.join(root, file))

    for jpeg_path in path_list:
        jpeg = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
        pic = crop_transform(picture_size)(torch.from_numpy(jpeg))
        jpeg_list.append(np.array(pic.div(pic.max())))
        name_list = jpeg_path.split('\\')
        for i in range(len(name_list)):
            if name_list[i] in label_name.keys():
                label_index = i
        label_list.append(label_name[name_list[label_index]])

    for asc_path, jpeg_path in zip(asc_path_list, path_list):
        asc_name = asc_path.split('\\')[-1].split('_ASC.jpg')[0]
        if setting == 'FUSAR':
            jpg_name = jpeg_path.split('\\')[-1].split('.png')[0]
        else:
            jpg_name = jpeg_path.split('\\')[-1].split('.JPG')[0]
        assert asc_name == jpg_name, "asc数据与图像数据名称不一致, 检查数据集"
        asc_jpeg = cv2.imread(asc_path, cv2.IMREAD_GRAYSCALE)
        pic = crop_transform(picture_size)(torch.from_numpy(asc_jpeg))
        asc_list.append(np.array(pic.div(pic.max())))

    jpeg_data = np.array(jpeg_list)
    asc_data = np.array(asc_list)
    jpeg_data = np.expand_dims(jpeg_data, axis=1)
    asc_data = np.expand_dims(asc_data, axis=1)
    # mask = sailency(data=data, id=id)
    label = np.array(label_list)
    data_set = torch.utils.data.TensorDataset(torch.from_numpy(jpeg_data).type(torch.FloatTensor),
                                              torch.from_numpy(asc_data).type(
                                                  torch.FloatTensor),
                                              torch.from_numpy(label).type(torch.LongTensor))

    return data_set, label_name
