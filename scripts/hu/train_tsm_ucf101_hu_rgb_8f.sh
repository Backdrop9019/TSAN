#target-only train

python main.py ucf101-hu hmdb51-hu RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
