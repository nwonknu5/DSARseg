import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import make_grid

from network_architecture.modules import VectorQuantizedVAE
from dataset import ProstateDataset, EyeDataset
from log import LoggerFactory

factory = LoggerFactory('vq-vae-train')
factory.create_logger()

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def train(data_loader, model, optimizer, args):
    loss_reconstruction_sum = 0
    loss_quantization_sum = 0
    loss_commit_sum = 0
    for images, _, _ in data_loader:
        images = images.to(args.device)
        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)

        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        loss_reconstruction_sum += loss_recons.item()
        loss_quantization_sum += loss_vq.item()
        loss_commit_sum += loss_commit.item()

        optimizer.step()
        args.steps += 1

    factory.logger.debug(
        ' reconstruction loss:' + str(loss_reconstruction_sum) +
        ' quantization loss:' + str(loss_quantization_sum) +
        ' Commitment loss:' + str(loss_commit_sum))

def test(data_loader, model, args, epoch, site_name):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _, names in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)
            # if (epoch+1) % 100 == 0:
            #     for i in range(0, x_tilde.shape[0]):
            #         image_save_path = './save/reconstructed/' + site_name + '/'
            #         torchvision.utils.save_image((images[i]), image_save_path + names[i] + '_img.png', padding=0)
            #         torchvision.utils.save_image((x_tilde[i]), image_save_path + names[i] + '.png', padding=0)
        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    factory.logger.info('training initial')

    if args.dataset == 'prostate':
        site_names = ['Task071_A_RUNMC', 'Task072_B_BMC', 'Task073_C_I2CVB', 'Task074_D_UCL', 'Task075_E_BIDMC',
                      'Task076_F_HK']
        num_channels = 1
    if args.dataset == 'eye':
        site_names = ['Task091_eye3', 'Task092_eye4', 'Task093_eye2', 'Task094_eye1']
        num_channels = 1

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for site_id in range(len(site_names)):
        site_name = site_names[site_id]
        save_filename = './save/models/'+ args.output_folder +'/'+str(site_name)+'/'
        image_save_path = './save/reconstructed/' + site_name + '/'
        encode_save_path = './save/z_e_x/' + site_name + '/'
        if not os.path.exists(save_filename):
            os.makedirs(save_filename)
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        # if not os.path.exists(encode_save_path):
        #     os.makedirs(encode_save_path)
        factory.logger.info('======================' + site_name + ' train =======================')
        factory.logger.info('training initial')

        train_dataset = ''
        if args.dataset == 'prostate':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            target_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_dataset = ProstateDataset(task_id=site_id, train=True, transform=transform, target_transform=target_transform)
            valid_dataset = ProstateDataset(task_id=site_id, train=True, transform=transform, target_transform=target_transform)

        if args.dataset == 'eye':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomResizedCrop(128),
            ])
            target_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_dataset = EyeDataset(task_id=site_id, train=True, transform=transform, target_transform=target_transform)
            valid_dataset = EyeDataset(task_id=site_id, train=True, transform=transform, target_transform=target_transform)


        # Define the data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers, pin_memory=True)

        best_loss = -1.
        for epoch in range(args.num_epochs):
            train(train_loader, model, optimizer, args)
            loss, _ = test(valid_loader, model, args, epoch+1, site_name)
            factory.logger.info('[epoch '+str(epoch)+'/' + str(args.num_epochs)+ '] traing loss: ' + str(loss))
            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
                with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                    torch.save(model.state_dict(), f)
            if (epoch+1) % 50 == 0:
                with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
                    torch.save(model.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=128,
        help='size of the latent vectors (default: 128)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=8,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./save/models'):
        os.makedirs('./save/models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    args.steps = 0

    main(args)
