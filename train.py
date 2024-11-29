from utils import *
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import *
from models import *

def main(config):
    set_seed()
    dm = DataModule(config)
    build_dirs(config)
    modelckpt = ModelCheckpoint(monitor='val_auroc',
                                dirpath=config['ckpt_dir'],
                                filename=config['file_name'],
                                mode='max')

    ckpt_dir = config['ckpt_dir_model']
    if os.path.exists(ckpt_dir):
        model = Trainer.load_from_checkpoint(ckpt_dir, config=config)
        print('load_weights')
    else:
        model = Trainer(config)

    model.train_dataloader_(dm.train_dataloader())

    num_epochs = config['num_epochs']
    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=[modelckpt],
        accelerator="gpu",
        logger=CSVLogger(config['output_log_dir']),
        check_val_every_n_epoch=1,
    )

    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':
    config = get_config()

    bb = ['b5']
    num_imgs_= [9000, 12000, 15000]
    for bb_ in bb:
        for num_imgs in num_imgs_:
            config['backbone'] = bb_
            config['num_imgs'] = num_imgs
            main(config)