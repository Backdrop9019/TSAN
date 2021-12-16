
# test TSM
python test_models.py hmdb51-uh \
    --weights=/data/gyeongho/framework/tsm_bp/checkpoint/uh/DBP/TSM_ucf101-uh_RGB_resnet50_shift8_blockres_avg_segment8_e30/ckpt.best.pth\
    --test_segments=8 --test_crops=1 \
    --test_list=/data/gyeongho/framework/tsm_bp/dataset/ufc-hmdb/hmdb_test_uh.txt \
    --batch_size=64 --threshold=True


