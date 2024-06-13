#dataset
# classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
#            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
data_root = './data/coco/'
#model
img_scale = (640, 640)
widen_factor=1.0
num_last_epochs=0
max_epochs=300
test_img_scale=(416,416)
pretrain_weight='./weights/femto_large_imagenet1k.pth'
load_from = './work_dirs/femtodet_large_coco_2stage/epoch_300.pth'
resume_from = None
#more information
optimizer = dict(
    type='SGD',
    lr=0.0001,
    momentum=0.90,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=num_last_epochs, priority=48),# all_unfreeze_after_this_epoch=50),
    dict(type='SyncNormHook', num_last_epochs=num_last_epochs, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
find_unused_parameters=True
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
default_channels=[32, 96, 320]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.82, 58.82, 58.82], to_rgb=True)
neck_in_chanels = [int(ch*widen_factor) for ch in default_channels]
headfeat_channel = 64
model = dict(
    type='FemtoDet',
    input_size=img_scale,
    random_size_range=(10, 20),
    random_size_interval=10,
    backbone=dict(
        type='FemtoNet',
        widen_factor=widen_factor,
        diff_conv=True,
        out_indices=(2, 4, 6),
        act_cfg=dict(type='ReLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=pretrain_weight)),
    neck=dict(
        type='SharedNeck',
        in_channels=neck_in_chanels,
        out_channels=headfeat_channel,
        fixed_size_idx=1,
        add=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        num_outs=1),
    bbox_head=dict(
        type='SharedDetHead',
        num_classes=80, #len(classes),
        in_channels=headfeat_channel,
        feat_channels=headfeat_channel,
        strides=[16,],
        stacked_convs=0,
        loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
        act_cfg=dict(type='ReLU'),
        use_depthwise=True),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
#data_root = 'data/coco/'
#dataset_type = 'CocoDataset'
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, prob=0.0),
    # dict(
    #     type='RandomAffine',
    #     superposition=False,
    #     scaling_ratio_range=(0.5, 1.5),
    #     border=(-img_scale[0]//2, -img_scale[1]//2)),
    # # dict(
    # #     type='FemtoDetMixUp',
    # #     superposition=False,
    # #     img_scale=img_scale,
    # #     ratio_range=(0.8, 1.2),
    # #     pad_val=114.0),
    # dict(type='YOLOXHSVRandomAug_EodVersion', hgain=0.015, sgain=0.7, vgain=0.4),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        # classes=classes,
        ann_file=data_root+'annotations/instances_train2017.json',
        img_prefix=data_root+'train2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=test_img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type='CocoDataset',
        # classes=classes,
        ann_file=data_root+'annotations/instances_val2017.json',
        img_prefix=data_root+'val2017',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        # classes=classes,
        ann_file=data_root+'annotations/instances_val2017.json',
        img_prefix=data_root+'val2017',
        pipeline=test_pipeline))

interval = 10
evaluation = dict(
    save_best='auto', interval=10, dynamic_intervals=[(285, 1)], metric='bbox')