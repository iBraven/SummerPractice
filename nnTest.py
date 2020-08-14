# Test Neural Network
import torch
import nnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def euclid(x, y):
    cor = x - y
    cor = torch.sum(cor.view(-1, 21, 3), 2)
    cor = cor * cor
    return cor


def num_correct(net, set):
    corect = 0
    total = 0
    with torch.no_grad():
        for data in set:
            images, points = data['image'].to(device), data['points'].to(device)
            output = net(images)

            total += points.size(0) * points.size(1)
            dist = euclid(points, output)
            corect += (dist < 1).sum()

        print(f"Accuracy: {100 * corect / total}")
        None

def test_model(test_set, net = None, path = None):
    test_net = nnModel.Net().to(device)
    if not (net or path):
        print('Fara argumente valide')
        return
    elif net:
        test_net = net
    else:
        test_net.load_state_dict(torch.load(path))
        test_net.eval()

    num_correct(test_net, test_set)

# a = torch.rand(1, 18)
# b = torch.rand_like(a)
# print(a, b)
# c = euclid(a, a+0.1)
# corect = (c<0.1).sum()
# print(c)
# print(corect)

