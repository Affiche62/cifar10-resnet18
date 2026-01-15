import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data import CIFAR10DataModule
from lit_module import CIFAR10LitModule
from config import Config


def main():
    data_module = CIFAR10DataModule(batch_size=Config.BATCH_SIZE)
    
    model = CIFAR10LitModule(learning_rate=Config.LEARNING_RATE)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=Config.CHECKPOINT_DIR,
        filename='cifar10-resnet18-{epoch:02d}-{val_acc:.4f}',
        save_top_k=1,
        mode='max'
    )
    
    logger = TensorBoardLogger(
        save_dir='logs/',
        name='cifar10_resnet18'
    )
    
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator=Config.NUM_GPUS,
        precision=Config.PRECISION,
        callbacks=[checkpoint_callback],
        logger=logger,
        deterministic=False,
        log_every_n_steps=10
    )
    
    trainer.fit(model, datamodule=data_module)
    
    test_results = trainer.test(model, datamodule=data_module)
    
    print(f"\nFinal Test Accuracy: {test_results[0]['test_acc']:.4f}")
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}")


if __name__ == '__main__':
    main()
