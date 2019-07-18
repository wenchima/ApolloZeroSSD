import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import torch.utils.data as data
import torch.backends.cudnn as cudnn



from ssd import build_ssd
from data import *
from data import coco_eval
from torch.utils.data import Dataset, DataLoader
from utils.augmentations import SSDAugmentation

from utils.logging import Logger


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main(args=None):

	parser = argparse.ArgumentParser(description='Simple test script for testing a SSD network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	#parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	#parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	#parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
	parser.add_argument('--save_file', default='save/', help='Directory for saving checkpoint models')
	parser.add_argument('--dir_mdl', default='./log/res-med/csv_med_imgnet_final.pt', help='Path to saved model')
	parser.add_argument('--num_workers', default=4, type=int,
	                    help='Number of workers used in dataloading')
	parser.add_argument('--cuda', default=True, type=str2bool,
	                    help='Use CUDA to train model')

	parser = parser.parse_args(args)


	#if not os.path.exists(parser.save_folder):
	#	os.mkdir(parser.save_folder)

	# first test
	sys.stdout = Logger(os.path.join(parser.save_file, 'log0.txt'))


	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_val = COCODetection(root=parser.coco_path,
								image_set='val2017',
                                transform=SSDAugmentation(coco['min_dim'],
                                                          MEANS))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	#sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
	#dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		#sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		#dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
		dataloader_val = data.DataLoader(dataset_val, 1,
	                                  num_workers=parser.num_workers,
	                                  shuffle=True, collate_fn=detection_collate,
	                                  pin_memory=True)

	ssd_net = build_ssd('test', coco['min_dim'], coco['num_classes'])            # initialize SSD
	ssd_net.load_state_dict(torch.load(parser.dir_mdl))

	if parser.cuda:
		ssd_net = ssd_net.cuda()
		cudnn.benchmark = True

	ssd_net.eval()
	print('Finished loading model!')

	if parser.dataset == 'coco':

		print('Evaluating dataset')

		coco_eval.evaluate_coco(dataset_val, ssd_net, parser.cuda, parser.save_file)

	elif parser.dataset == 'csv' and parser.csv_val is not None:

		print('Evaluating dataset')

		mAP = csv_eval.evaluate(dataset_val, retinanet)
	#retinanet.eval()

	#torch.save(retinanet, parser.save_folder + '/model_final_v2.pt')

if __name__ == '__main__':
 main()
