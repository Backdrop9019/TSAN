# test w treshold
python test_models.py ucf101-hu \
    --weights=/data/gyeongho/framework/tsm_bp/checkpoint/hu/UBP/TSM_hmdb51-hu_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.best.pth\
    --test_segments=8 --test_crops=1 \
    --test_list=/data/gyeongho/framework/tsm_bp/dataset/hmdb-ufc/ucf101_test_hu.txt\
    --batch_size=64 --threshold=

