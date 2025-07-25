import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torch.autograd import Function

class Decoder(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim,in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim,in_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class AdverserialLoss(nn.Module): 
    def __init__(self, num_classes, args,temperature = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.y_t = torch.tensor([1/num_classes for _ in range(num_classes)])
        self.temperature = temperature
        self.args = args


    def forward(self, source_entropy, target_entropy, labelled_or_not, labels, disc = True):
        y_source = labels[labelled_or_not]
        source_loss = nn.CrossEntropyLoss()(source_entropy, y_source)

        target_entropy = F.softmax(target_entropy/self.temperature, dim = 1)
        y_target = self.y_t.expand(target_entropy.shape[0],-1).to(self.args.device)
        # maxi,_ = torch.max(entropy[~labelled_or_not], dim = 0) # for normalizing

        target_loss = torch.sum(-y_target*torch.log(target_entropy)) 

        # print(source_loss, target_loss)
        if disc:
            return source_loss + target_loss
        else:
            return source_loss - target_loss

class MarginLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
        self.l1 = nn.L1Loss()

    def forward(self, sim, diff,  gt):
        loss = self.l1(sim,gt) - self.l1(diff,gt)
        return max(torch.tensor(0), loss)


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd).apply(x)

class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.out_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(output_dim, output_dim//2),
            nn.LeakyReLU(),
            nn.Linear(output_dim//2, output_dim//4),
            nn.LeakyReLU(),
            nn.Linear(output_dim//4, output_dim//8),
            nn.LeakyReLU(),
            nn.Linear(output_dim//8, output_dim//4),
            nn.LeakyReLU(),
            nn.Linear(output_dim//4,output_dim//2),
            nn.LeakyReLU(),
            nn.Linear(output_dim//2,output_dim),
            nn.LeakyReLU(),
        )
        self.lambd = 1

    def set_lambda(self, lambd=1):
        self.lambd = lambd

    def forward(self, x,reverse = False):
        if reverse:
            x = grad_reverse(x, self.lambd)
            out = self.network(x)
        else:
            out = self.network(x)
            
        return out



def gram_matrix(feat):
    # https://github.com/pytorch/ex`amples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class ReconstructionLoss(nn.Module):
    def __init__(self, args, extractor=VGG16FeatureExtractor()):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.args = args

    def forward(self, mask, output, gt):
        # loss = torch.tensor(0.0)
        loss_dict = {}
        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)
        loss_dict['prc'] = torch.tensor(0.0).to(self.args.device)
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
        loss_dict['style'] = torch.tensor(0.0).to(self.args.device)
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        # loss_dict['tv'] = total_variation_loss(output_comp)
        # for k,v in loss_dict.items():
        #     loss+=v
        
        return torch.sum(torch.stack(list(loss_dict.values())))
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)
        negative_distance = (anchor - negative).pow(2).sum(1)
        loss = torch.relu(positive_distance - negative_distance + self.margin)
        return loss.mean()