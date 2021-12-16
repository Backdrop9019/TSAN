# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
python main.py ucf101-uh hmdb51-uh RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.002 --wd 1e-4 --lr_steps 20 40 --epochs 30 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=5 \
     --shift --shift_div=8 --shift_place=blockres --npb --da=DBP