# import torch
# import matplotlib.pyplot as plt
# import nnModel
# import LoadData
#
# device = ('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True
# # a = torch.rand(2, 32, 1)
# # print(a.numel())
# net = nnModel.Net()
# path = 'D:\\Model\\Model1.pth'
# check = torch.load(path)
# net.load_state_dict(check)
# net.to(device)
# net.eval()
#
# train_loader, test_loader = LoadData.load_data(LoadData.img_path, nr_rows=1600, transform=LoadData.img_transform, batch_size=16)
# it = iter(train_loader)
# data = next(it)
# with torch.no_grad():
#     while data:
#         img = data['image'].to(device)
#         points = data['points'].to(device)
#         out = net(img.reshape(-1, 1, 128, 128))
#         out = (out + 1) * 64
#         point = LoadData.denormalize_points(points)
#         # print(f'out: {out}')
#         # print(f'points: {point}')
#         # out = torch.squeeze(out, dim=0)
#         for i in range(len(img)):
#             plt.subplot(4, 4, i + 1)
#             plt.imshow(img[i].cpu().reshape(128, 128))
#             plt.scatter(out[i][::2].cpu(), out[i][1::2].cpu(), s=9, c='blue')
#             plt.scatter(point[i][::2].cpu(), point[i][1::2].cpu(), s=9, c='red')
#         plt.show()
#         data = next(it)
