import os
import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision.transforms as T
import matplotlib.pyplot as plt
import tqdm
from model import G, D


dataset_dir = "cifar10/"
batch_size = 32
learning_rate = 1e-4
num_epochs = 200
checkpoints_dir = 'checkpoints/'
clamp_num = 0.01
num_workers = 16


transforms = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

train_dataset = torchvision.datasets.CIFAR10(
        root = dataset_dir,
        train = True, # True: training data; False: testing data
        transform = transforms, # ndarray => torch tensor
        download = False, # whether download or not
        )

train_dataloader = Data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

noises = torch.randn(batch_size, 100, 1, 1)
noises = noises.cuda()
fix_noises = torch.randn(batch_size, 100, 1, 1)
fix_noises = fix_noises.cuda()


# G, D
def init_weights(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)

G.apply(init_weights)
D.apply(init_weights)

G = G.cuda()
D = D.cuda()

if os.path.exists(os.path.join(checkpoints_dir, 'G.pkl')):
    print('loading G...')
    G = torch.load(os.path.join(checkpoints_dir, 'G.pkl'))
if os.path.exists(os.path.join(checkpoints_dir, 'D.pkl')):
    print('loading D...')
    D = torch.load(os.path.join(checkpoints_dir, 'D.pkl'))

G.train()
D.train()


# loss
# optimizer
G_optimizer = torch.optim.RMSprop(G.parameters(), lr=learning_rate)
D_optimizer = torch.optim.RMSprop(D.parameters(), lr=learning_rate)

ones = torch.ones(batch_size, 1, 1, 1) 
nones = -1 * torch.ones(batch_size, 1, 1, 1) 
ones = ones.cuda()
nones = nones.cuda()

for epoch in range(num_epochs):
    for index, (image, _) in tqdm.tqdm(enumerate(train_dataloader)): # train_data is a list with length 2. [image data, image label]
        real_image = Variable(image)
        real_image = real_image.cuda()


        for param in D.parameters():
            param.data.clamp_(-clamp_num, clamp_num)

        #####################################################################
        # Discriminator
        #####################################################################
        if index % 1 == 0:
            D_optimizer.zero_grad()
            
            real_pred = D(real_image)
            D_loss_real = torch.sum(real_pred)
            #D_loss_real.backward(nones)

            noises.data.normal_()
            fake_image = G(noises).detach() # fake_image.requires_grad => False
            fake_pred = D(fake_image)
            D_loss_fake = -1 * torch.sum(fake_pred)
            #D_loss_fake.backward(ones)

            #D_loss = (D_loss_fake + D_loss_real) / batch_size
            D_loss = D_loss_fake + D_loss_real
            

            #D_loss = D_loss_real + D_loss_fake
            #D_loss.backward()
            D_loss.backward()
            D_optimizer.step()

        #####################################################################
        # Generator
        #####################################################################
        if index % 5 == 0: 
            G_optimizer.zero_grad()
            
            noises.data.normal_()
            fake_image = G(noises)
            fake_pred = D(fake_image)
            #G_loss = torch.sum(fake_pred) / batch_size
            G_loss = torch.sum(fake_pred)
            #G_loss.backward(nones)
            G_loss.backward()
            G_optimizer.step()
            
        if index % 10 == 0:
            print('epoch {}: D loss={}; G loss={}'.format(epoch, D_loss, G_loss))
    
    G.eval()
    D.eval()
    print('saving...')
    fake_image = G(fix_noises)
    scores = D(fake_image).detach()
    scores = scores.data.squeeze()
    indexs = scores.topk(32)[1]
    result = list()
    for index in indexs:
        result.append(fake_image.data[index])

    torchvision.utils.save_image(torch.stack(result), 'figures/result_{}.png'.format(epoch), normalize=True, range=(-1, 1))
    G.train()
    D.train()
    
    #####################################################################
    # Save Model
    #####################################################################
    torch.save(G, 'checkpoints/G.pkl')
    torch.save(D, 'checkpoints/D.pkl')

