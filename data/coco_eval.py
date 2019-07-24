from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os
import time

import torch
from torch.autograd import Variable


from data.config import coco
#from data.coco import COCO_CLASSES

import pdb



class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def evaluate_coco(dataset, model, is_cuda, save_folder, threshold=0.05):

    model.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []


        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(len(dataset))]
                     for _ in range(coco['num_classes'])]

        # timers
        _t = {'im_detect': Timer(), 'misc': Timer()}

        # for test, kaidong
        #pdb.set_trace()

        # for test, kaidong
        #for index in range(60):
        for index in range(len(dataset)):

            img, gt, h, w = dataset.pull_item(index)
            #data = dataset[index]

            x = Variable(img.unsqueeze(0))

            if is_cuda:
                x = x.cuda()

            _t['im_detect'].tic()
            detections = model(x).data
            detect_time = _t['im_detect'].toc(average=False)

            for j in range(1, detections.size(1)):
                det = detections[0, j]
                index_valid = det[:, 0].gt(0.)

                if index_valid.sum() < 1:
                    continue

                det = det[index_valid]

                det[:, 1] *= w
                det[:, 3] *= w
                det[:, 2] *= h
                det[:, 4] *= h

                # change to coco format, kaidong
                det[:, 3] -= det[:, 1]
                det[:, 4] -= det[:, 2]

                for k in range(det.size(0)):
                    score = det[k, 0]
                    box = det[k, 1:]

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.ids[index],
                        'category_id' : dataset.target_transform.inverse_label_map[j],
                        #'category_id' : COCO_CLASSES[j-1],
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }


                    # for test, kaidong
                    #if score > 0.5:
                    #    print('coco eval', 'res', image_result)


                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.ids[index])

            print('im_detect: {:d}/{:d} {:.3f}s'.format(index + 1, len(dataset), detect_time))



        if not len(results):
            return

        file_json = os.path.join(save_folder, '{}_bbox_results.json'.format(dataset.img_set))

        # write output
        json.dump(results, open(file_json, 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(file_json)

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return
