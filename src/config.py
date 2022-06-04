class config:
    random_seed = 9898

    batch_size = 64
    split_ratio = 0.8
    use_shuffle = True
    img_dim = (256, 256)

    base_filter_size = 64

    weight_decay = 1e-3
    lr = 1e-4
    epochs = 10
    log_interval = 5
