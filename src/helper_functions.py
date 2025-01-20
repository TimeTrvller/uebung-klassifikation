import os
import numpy as np


def create_colored_point_cloud(point_cloud, predictions):
    '''
    Creates color encoded point cloud based on the predictions and saves it as filename.ply 
    Input:
        point_cloud: N x 4 point cloud with X, Y, Z coordinates and class labels (numpy array)
        predictions: N x 1 vector of the random forest predictions with values from 1 to 5 representing the five classes (numpy array)
    Output:
        colored_point_cloud: color encoded point cloud of the shape N x [X, Y, Z, R, G, B] (numpy array)
    '''

    # Sanity check
    assert len(point_cloud) == len(predictions)

    # dataset_name = 'Oakland_3D_Point_Cloud_Dataset_5Classes'

    # Class RGB values
    classes_rgb = np.array([
    [0, 0, 255],            # wire
    [255, 0, 0],            # pole/trunk
    [200, 200, 200],        # facade
    [255, 190, 0],          # ground
    [0, 255, 65]            # vegetation
    ], dtype='u1')

    colored_predictions = np.zeros((len(predictions), 3))
    for c in range(0, 5):
        colored_predictions[predictions == c+1] = classes_rgb[c]

    colored_point_cloud = np.empty((len(predictions), 6)) 
    colored_point_cloud[:, :3] = point_cloud[:, :3]
    colored_point_cloud[:, 3:] = colored_predictions

    # print(colored_point_cloud.shape, colored_point_cloud)

    return colored_point_cloud  # shape N x [X, Y, Z, R, G, B]


def save_colored_point_cloud_as_ply(colored_point_cloud, file_addition: str):
    '''
    Saves a color encoded point cloud as *.ply in the current working directory
    Input:
        colored_point_cloud: a numyp array of the color encoded point cloud of the shape N x [X, Y, Z, R, G, B]
        file_addition: a string specifying the file name
    '''

    n = len(colored_point_cloud)
    max_size = np.max(colored_point_cloud.shape)

    current_dir = f"{os.getcwd()}/data"
    file_name = current_dir + '/point_cloud_' + file_addition + '.ply'

    with open(file_name, 'w') as writer:
        writer.write('ply\n')
        writer.write('format ascii 1.0\n')
        writer.write(f'element vertex {max_size:7.0f}\n')
        writer.write('property float x\n')
        writer.write('property float y\n')
        writer.write('property float z\n')
        writer.write('property uchar diffuse_red\n')
        writer.write('property uchar diffuse_green\n')
        writer.write('property uchar diffuse_blue\n')
        writer.write('end_header\n')

        for i in range(n):
            y = colored_point_cloud[i, :6]
            writer.write(f'{y[0]:4.4f} {y[1]:4.4f} {y[2]:4.4f} {y[3]:3.0f} {y[4]:3.0f} {y[5]:3.0f}\n')
        
        writer.write('\n')
