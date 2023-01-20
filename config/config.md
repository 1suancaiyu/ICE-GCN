## train

### ntu 120 csub bone 
origin:
add SCE:
```
--config config/nturgbd120-cross-subject/default.yaml --train-feeder-args bone=True --test-feeder-args bone=True --model model.ctrgcn_sce.Model --weights ./pretrained_model/CTRGCN_NTU120_CSub_bone_85.7/runs-60-59040.pt --work-dir work_dir/ntu120/SCE/0506_ctrgcn_sce_ntu120_csub_bone --device 0 --num-epoch 150 --batch-size 64 --test-batch-size 64
```

### ntu60 csub joint
origin:
add TCE
```
--config config/nturgbd-cross-subject/default.yaml --model model.ctrgcn_tce.Model --work-dir work_dir/ntu60/0507_cs_joint_ctrgcn_tce --device 1 --num-epoch 150 --batch-size 64 --test-batch-size 64
```

## Testing
### pretrained models

ntu120 csub bone
85.7%
```
python main.py --config ./config/nturgbd120-cross-subject/default.yaml  --test-feeder-args bone=True --work-dir work_dir/test_ntu120_csub_bone_ctrgcn --phase test --save-score True --weights pretrained_model/CTRGCN_NTU120_CSub_bone_85.7/runs-60-59040.pt --device 0
```

