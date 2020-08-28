# Test Neural Network
import torch

import LoadData
import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_correct(net, test_set):
    with torch.no_grad():
        for i, data in enumerate(test_set, 0):
            # Data to device (gpu)
            images, points = data['image'].to(device), data['points'].to(device)
            output = net(images)
            LoadData.show_batch({'image': images.to('cpu'), 'points': points.to('cpu')}, output.to('cpu'))
            if i == 5:
                break


def test_model(test_set, net=None, path=None):
    test_net = Model.Net().to(device)
    if not (net or path):
        print('Fara argumente valide')
        return
    elif net:
        test_net = net
    else:
        test_net.load_state_dict(torch.load(path))
        # reteaua trece in modul eval pentru test
        test_net.eval()

    num_correct(test_net, test_set)
