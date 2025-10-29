# Step1, generate 3d coordinate for iname database


```
python gen3d_write_lmdb.py --sdf /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/Enamine_screening_collection_202508.sdf \
  --lmdb /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/Enamine_202508_3d.lmdb \
  --map-size-gb 64 \
  --batch-commit 2000
```

# Step2, extract molecular features for iname database


```
python -u unimol/test.py --user-dir ./unimol ./data --valid-subset test --results-path ./test --num-workers 8 --ddp-backend c10d --batch-size 8 --task drugclip --loss in_batch_softmax --arch drugclip --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 --path checkpoint_best.pt --log-interval 100 --log-format simple --max-pocket-atoms 511 --test-task extract_feats --mol_data_path /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/Enamine_202508_3d.lmdb --save_dir /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/mol_feats
```



# Step3, extract pocket information


```
python -u /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/extract_pocket.py
```



# step4,  retrival and get the result


```
python -u unimol/test.py --user-dir ./unimol ./data --valid-subset test --results-path ./test --num-workers 8 --ddp-backend c10d --batch-size 8 --task drugclip --loss in_batch_softmax --arch drugclip --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 --path checkpoint_best.pt --log-interval 100 --log-format simple --max-pocket-atoms 511 --test-task retrival  --molemb_path /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/mol_feats/Enamine_202508_3d.lmdb.pkl --pock_path /vepfs-mlp2/mlp-public/shikunfeng/Project/DrugCLIP/data/inami/pocket_data.lmdb
```