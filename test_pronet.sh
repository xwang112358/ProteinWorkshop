CUDA_VISIBLE_DEVICES=0 python proteinworkshop/train.py \
  dataset=ec_random \
  task=multiclass_graph_classification \
  encoder=pronet \
  features=pronet_backbone \
  trainer.max_epochs=1 \
  test=True \
  hydra.run.dir=outputs/train-pronet-test/ec_random/pronet