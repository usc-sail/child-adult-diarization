import shutil, argparse, os, yaml, torch, random
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from lightning_modules.classifier import Classifier


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    # parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--debug", type=str, default="t")
    arg_parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = arg_parser.parse_args()

    # load configs
    with open(args.config, "r") as yaml_file:
        configs = yaml.safe_load(yaml_file)
    if args.debug == "t":
        configs["debug"] = True
    else:
        configs["debug"] = False
    
    # set seed
    set_seed(configs["seed"])    

    # remove previous checkpoints
    checkpoint_path = os.path.join(configs["save"], "checkpoints", configs["name"])
    if os.path.exists(checkpoint_path):
        print("Removing previous checkpoints: ", checkpoint_path)
        import pdb; pdb.set_trace()
        shutil.rmtree(checkpoint_path)

    # train
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=False, 
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        dirpath=checkpoint_path,
        filename="{epoch}-{val_loss:.3f}",
        every_n_train_steps = 0,
        every_n_epochs = 1
    )

    logger = TensorBoardLogger("logs", name=configs["name"])

    trainer = pl.Trainer(logger = logger, callbacks=[early_stop_callback, checkpoint_callback], 
                            # gradient accumulation
                            accumulate_grad_batches=configs["accumulate_grad_batches"],
                            devices=configs["devices"], 
                            accelerator="gpu", 
                            max_epochs=configs["max_epochs"])

    
    model = Classifier(
        hparams=configs,
    )
    
    trainer.fit(model)
    print("best model: ", trainer.checkpoint_callback.best_model_path)
    model_test = Classifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
    
    result = trainer.test(model_test)
    # print(result)