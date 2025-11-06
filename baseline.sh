python proteinworkshop/train.py dataset=ec_structure task=multiclass_graph_classification \
encoder=gvp features=all_equivariant_ca trainer.max_epochs=1 test=True \
hydra.run.dir='outputs/train/ec_structure'


