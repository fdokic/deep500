#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:56:12 2019

@author: fdokic
adapted from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.utils.data

from torch.autograd import Variable



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
class DCGan(nn.Module):
    def __init__(self, nz, ngf, nc, ndf, b_size):
        """ Creates DCGan object
            @param nz size of latent vector for generator
            @param ngf Size of feature maps in generator
            @param nc number of color channels of input
            @ndf Size of feature maps in discriminator
        """
        super(DCGan, self).__init__()
        self.netD = Discriminator(nc, ndf)
        self.netG = Generator(nz, ngf,nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ndf = ndf
        self.b_size = b_size
        
    def forward(self, input, noise):
        
        # noise = torch.randn(self.b_size, self.nz, 1, 1, device='cpu')
        return self.netD.forward(input), self.netG.forward(noise)
       
    
def export_dcgan(nz=100, ngf=64, nc=3, ndf=64, b_size=128, shape=(3, 64, 64), file_path='DCGan.onnx') -> str:
    model = DCGan(nz, ngf, nc, ndf, b_size)
    dummy_input = Variable(torch.randn(b_size, *shape))
    dummy_noise = Variable(torch.randn(model.b_size, model.nz, 1, 1))
    input = (dummy_input, dummy_noise)
    torch.onnx.export(model, input, file_path, verbose=False,
                      training=True)
    return file_path

def export_train_dcgan(nz=100, ngf=64, nc=3, ndf=64, b_size=128, shape=(3, 64, 64), file_path='DCGan.onnx') -> str:
    path = export_dcgan(nz, ngf, nc, ndf, b_size, shape, file_path=file_path)
    return dcgan_inference_to_training(path)


def export_discriminator(nc=3, ndf=64, b_size=128, shape=(3, 64, 64), file_path='DCGan_D.onnx') -> str:
    model = Discriminator(nc, ndf)
    dummy_input = Variable(torch.randn(b_size, *shape))
    
    torch.onnx.export(model, dummy_input, file_path, verbose=False,
                      training=True)
    return file_path


def export_generator(nz=100, ngf=64, nc=3, b_size=128, file_path='DCGan_G.onnx') -> str:
    shape = (nz, 1, 1)
    model = Generator(nz, ngf, nc)
    dummy_input = Variable(torch.randn(b_size, *shape))
    
    torch.onnx.export(model, dummy_input, file_path, verbose=False,
                      training=True)
    return file_path

def dcgan_inference_to_training(path: str, export_init_graph=False):
    import onnx
    import deep500.tools.initialization_graphs as init
    import deep500.tools.input_completion as completion
    from onnx import TrainingInfoProto, StringStringEntryProto, ModelProto

    model = onnx.load(path)
    initializer_graph = init.InitializationGraph()

    compensated_weights = []
    for node in model.graph.node:
        if node.op_type == 'Conv' or node.op_type == 'ConvTranspose':
            weight_info = [(w.name, w.dims, w.data_type) for w in model.graph.initializer if w.name in node.input].pop()
            initializer_graph.add_node('RandomNormal', weight_info[1], (0.0, 0.02),
                                       weight_info[0], weight_info[2])
            compensated_weights.append(weight_info[0])
        if node.op_type == 'BatchNormalization':
            for name in node.input:
                if 'weight' in name:
                    weight_info = [(w.name, w.dims, w.data_type) for w in model.graph.initializer if
                                   w.name == name].pop()
                    initializer_graph.add_node('RandomNormal', weight_info[1], (1.0, 0.02), weight_info[0],
                                               weight_info[2])
                    compensated_weights.append(name)
                if 'bias' in name or 'running_mean' in name:
                    pass
                    # weight_info = [(w.name, w.dims, w.data_type) for w in model.graph.initializer if
                    #                w.name == name].pop()
                    # initializer_graph.add_node('ConstantOfShape', weight_info[1], 0, name, weight_info[2])
                    # compensated_weights.append(name)
                # if 'running_mean' in name:
                if 'running_var' in name:
                    pass
                    # weight_info = [(w.name, w.dims, w.data_type) for w in model.graph.initializer if
                    #                w.name == name].pop()
                    # initializer_graph.add_node('ConstantOfShape', weight_info[1], 1, name, weight_info[2])
                    # compensated_weights.append(name)

    onnx_init_graph = initializer_graph.make_graph()

    for w in [w for w in model.graph.initializer if w.name in compensated_weights]:
        model.graph.initializer.remove(w)

    new_model = ModelProto()
    new_model.CopyFrom(model)
    new_model.graph.CopyFrom(model.graph)
    new_model.graph.input.extend(model.graph.input)

    bindings = initializer_graph.get_bindings()
    onnx_bindings = []
    for k in bindings.keys():
        b = StringStringEntryProto()
        b.value = bindings[k]
        b.key = k
        onnx_bindings.append(b)

    train = TrainingInfoProto()
    train.initialization.CopyFrom(onnx_init_graph)

    train.initialization_binding.extend(onnx_bindings)

    new_model.training_info.extend([train])
    new_path = path[:-5] + '_train.onnx'
    new_model = completion.complete_inputs(new_model)
    onnx.save(new_model, new_path)

    if export_init_graph:
        # for demo-purpose: save init graph separately for e.g. netron
        demo = onnx.helper.make_model(onnx_init_graph)
        onnx.save(demo, 'dcgan_init_demo.onnx')

    return new_path


