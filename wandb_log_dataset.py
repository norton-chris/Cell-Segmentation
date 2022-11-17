import wandb

run = wandb.init(project='Cell-Segmentation', entity="nort")
artifact = wandb.Artifact('Cells', type='dataset')
artifact.add_dir("TrainingDataset/data_subset/323_subset/output/")  # Adds multiple files to artifact
run.log_artifact(artifact)