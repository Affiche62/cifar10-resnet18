class Config:
    BATCH_SIZE = 128
    MAX_EPOCHS = 50
    NUM_WORKERS = 4
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    DATA_DIR = './data'
    CHECKPOINT_DIR = './checkpoints'
    
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2023, 0.1994, 0.2010]
    
    RANDOM_CROP_PADDING = 4
    RANDOM_FLIP_PROB = 0.5
    
    NUM_CLASSES = 10
    NUM_GPUS = 'auto'
    
    PRECISION = '16-mixed'
