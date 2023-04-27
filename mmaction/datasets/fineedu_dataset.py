# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import time
import os.path as osp
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_map, eval_recalls

from ..core.evaluation.ava_utils import ava_eval, read_labelmap, results2csv
from ..core.evaluation.ava_evaluation import object_detection_evaluation as det_eval
from ..core.evaluation.ava_evaluation import standard_fields
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class FineEduDataset(BaseDataset):
    """FineEdu dataset for spatial temporal detection.

    Based on FineEdu annotation files, the dataset loads raw frames,
    bounding boxes and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> annotation.json

    Args:
        ann_file (str): Path to the annotation file like
            ``annotation.json``.
        pipeline (list[dict | callable]): A sequence of data transforms.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        proposal_file (str): Path to the proposal file like
            ``ava_proposal.pkl``.
            Default: None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used. Default: 0.9.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used.
        num_classes (int): The number of classes of the dataset. Default: xxx+1.
            (FineEdu has xxx action classes, another 1-dim is added for potential
            usage)
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                        Default: 'RGB'.
        num_max_proposals (int): Max proposals number to store. Default: 1000.
        fps (int): Overrides the default FPS for the dataset. Default: 24.
    """
    POSE_CLASSES = ['lower the head', 'turn aside', 'lift the head', 'stand up', 'turn around', 'walk']
    ACTION_CLASSES = ['read', 'small action', 'play phone', 'look student', 'look teacher', 'unknown', 'talk', 'answer the question', 'other', 'look blackboard', 'distracted', 'write', 'photograph', 'yawn', 'play computer', 'sleep', 'eat', 'drink', 'stretch']
    CLASSES = ACTION_CLASSES

    def __init__(self,
                 ann_file,
                 pipeline,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_classes=20,
                 custom_classes=None,
                 data_prefix=None,
                 test_mode=False,
                 modality='RGB',
                 num_max_proposals=1000,
                 fps=24):
        # since it inherits from `BaseDataset`, some arguments
        # should be assigned before performing `load_annotations()`
        self._FPS = fps  # Keep this as standard
        self.custom_classes = custom_classes
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = tuple([0] + custom_classes)
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.num_classes = len(self.CLASSES) + 1
        self.filename_tmpl = filename_tmpl
        self.num_max_proposals = num_max_proposals
        self.generate_class_to_index()
        self.logger = get_root_logger()
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            start_index=start_index,
            modality=modality,
            num_classes=num_classes)

        if self.proposal_file is not None:
            self.proposals = mmcv.load(self.proposal_file)
        else:
            self.proposals = None
            
    def generate_class_to_index(self):
        self.category2index = {category: i for i, category in enumerate(self.CLASSES)}

    def load_annotations(self):
        """Load FineEdu annotations."""  
        video_infos = mmcv.load(self.ann_file)
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_id = results['image_Id']
        
        frame_dir = results['video_info']
        if self.data_prefix is not None:
            frame_dir = osp.join(self.data_prefix, frame_dir)
        results['frame_dir'] = frame_dir

        gt_bboxes = []
        gt_labels = []                                                                
        for j, ann_info in enumerate(results['persons']):
            gt_bboxes.append(ann_info['person_bbox'])
            one_hot_label = np.zeros(self.num_classes, dtype=np.float32)
            #one_hot_label[self.category2index[ann_info['pose_id']]] = 1.
            for k, action_info in enumerate(ann_info['action_id']):
                one_hot_label[self.category2index[action_info]] = 1.
            gt_labels.append(one_hot_label)
        gt_bboxes = np.stack(gt_bboxes).astype(np.float32)
        gt_labels = np.stack(gt_labels)
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = gt_labels
        results['proposals'] = gt_bboxes
        results['scores'] = np.array([1])

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['fps'] = self._FPS

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results['image_Id']
        
        frame_dir = results['video_info']
        if self.data_prefix is not None:
            frame_dir = osp.join(self.data_prefix, frame_dir)
        results['frame_dir'] = frame_dir

        gt_bboxes = []
        gt_labels = []                                                                
        for j, ann_info in enumerate(results['persons']):
            gt_bboxes.append(ann_info['person_bbox'])
            one_hot_label = np.zeros(self.num_classes, dtype=np.float32)
            #one_hot_label[self.category2index[ann_info['pose_id']]] = 1.
            for k, action_info in enumerate(ann_info['action_id']):
                one_hot_label[self.category2index[action_info]] = 1.
            gt_labels.append(one_hot_label)
        gt_bboxes = np.stack(gt_bboxes).astype(np.float32)
        gt_labels = np.stack(gt_labels)
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = gt_labels
        results['proposals'] = gt_bboxes
        results['scores'] = np.array([1])

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['fps'] = self._FPS

        return self.pipeline(results)

    def dump_results(self, results, out):
        """Dump predictions into a csv file."""
        assert out.endswith('csv')
        results2csv(self, results, out, self.custom_classes)
        
    def gt_to_eval(self):
        """Convert groundtruth to the format of evaluation"""
        
        gt_bboxes = defaultdict(list)
        gt_labels = defaultdict(list)   
        for i, img_info in enumerate(self.video_infos):
            img_w = img_info['width']
            img_h = img_info['height']
            for j, ann_info in enumerate(img_info['persons']):
                bboxes = np.array(ann_info['person_bbox'], dtype=float)
                bboxes[0::2] /= img_w
                bboxes[1::2] /= img_h
                #gt_bboxes[img_info['image_Id']].append(bboxes)
                #gt_labels[img_info['image_Id']].append(self.category2index[ann_info['pose_id']] + 1)
                for k, action_info in enumerate(ann_info['action_id']):
                    gt_bboxes[img_info['image_Id']].append(bboxes)
                    gt_labels[img_info['image_Id']].append(self.category2index[action_info] + 1)
            
        return gt_bboxes, gt_labels
    
    def results_to_eval(self, results):
        """Convert results to the format of evaluation"""
        boxes = defaultdict(list)
        labels = defaultdict(list)
        scores = defaultdict(list)
        for idx in range(len(self.video_infos)):
            image_Id = self.video_infos[idx]['image_Id']
            result = results[idx]
            for label, _ in enumerate(result):
                for bbox in result[label]:
                    bbox_ = tuple(bbox.tolist())
                    actual_label = label + 1
                    boxes[image_Id].append(bbox_[:4])
                    labels[image_Id].append(actual_label)
                    scores[image_Id].appesnd(bbox_[4])

        return boxes, labels, scores
            
    def print_time(self, message, start):
        print('==> %g seconds to %s' % (time.time() - start, message), flush=True)
        
    def get_anno_eval(self):
        """Get the annotations for VOC style annotations."""                
        annotations = []
        for i, img_info in enumerate(self.video_infos):
            img_w = img_info['width']
            img_h = img_info['height']
            gt_bboxes = []
            gt_labels = []
            for j, ann_info in enumerate(img_info['persons']):
                bboxes = np.array(ann_info['person_bbox'], dtype=float)
                bboxes[0::2] /= img_w
                bboxes[1::2] /= img_h
                #gt_bboxes.append(np.expand_dims(bboxes, axis=0))
                #gt_labels.append(np.expand_dims(self.category2index[ann_info['pose_id']], axis=0))
                for k, action_info in enumerate(ann_info['action_id']):
                    gt_bboxes.append(np.expand_dims(bboxes, axis=0))
                    gt_labels.append(np.expand_dims(self.category2index[action_info], axis=0))
            gt_bboxes = np.concatenate(gt_bboxes, axis = 0)
            gt_labels = np.concatenate(gt_labels, axis = 0)
            print(gt_bboxes.shape , gt_labels.shape)
            
            img_info = {
                'bboxes': gt_bboxes,
                'labels': gt_labels
            }
            annotations.append(img_info)
        return annotations

    def evaluate_voc(self,
                results,
                metric='mAP',
                logger=None,
                iou_thr=0.5,):
        """Evaluate the prediction results and report mAP according VOC dataset."""
        
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        #annotations = [self.get_ann_info(i) for i in range(len(self))]
        annotations = self.get_anno_eval()
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            ds_name = 'ilsvrc_det'
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset=ds_name,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
        
    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):
        """Evaluate the prediction results and report mAP."""   
        
        mmcv.dump(self.video_infos, './video_infos.pkl')
        mmcv.dump(results, './results.pkl')
        print('dump complete')
        
        assert len(metrics) == 1 and metrics[0] == 'mAP', (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            'See https://github.com/open-mmlab/mmaction2/pull/567 '
            'for more info.')
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        labelmap = [{'id': i+1, 'name': category} for i, category in enumerate(self.CLASSES)]
        # Evaluation for mAP
        pascal_evaluator = det_eval.PascalDetectionEvaluator(labelmap)
        
        gt_boxes, gt_labels = self.gt_to_eval()
        boxes, labels, scores = self.results_to_eval(results)
        
        #print(gt_labels, labels)
        #for image_key in gt_labels:
        #    print(scores[image_key])
        
        for image_key in gt_boxes:
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(gt_boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                    np.array(gt_labels[image_key], dtype=int)
                })
        for image_key in boxes:
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float)
                })
            
        start = time.time()
        eval_result = pascal_evaluator.evaluate()
        self.print_time('run_evaluator', start)
        
        ret = {}
        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            log_msg = []
            for k, v in eval_result.items():
                log_msg.append(f'\n{k}\t{v: .4f}')
            log_msg = ''.join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)

        return ret
