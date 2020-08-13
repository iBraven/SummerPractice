# Test Neural Network
import torch
import nnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def num_correct(net, set):
    corect = 0
    total = 0
    with torch.no_grad():
        for data in set:
            images, points = data
            output = net(images)

            total = points.size(0)
            corect += (output == points).sum().item()
        print(f"Accuracy: {100 * corect / total}")

def test_model(test_set, net = None, path = None):
    test_net = nnModel.Net()
    if not (net or path):
        print('Fara argumente valide')
        return
    elif net:
        test_net = net
    else:
        test_net.load_state_dict(torch.load(path))
        test_net.eval()

    test_net.to(device)
    num_correct(test_net, test_set)
