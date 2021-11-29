class ConfigCLASS:
    def __init__(self):
        self.add_noise = False
        self.noise_variance = 0.04
        self.data_root_path = '../GridCellDataset/npy'
        self.check_point_path = './checkpoint_v17'
        self.random_seed = 1
        # self.n_place_cells = 256
        self.dropout_rate = 0.3
        # self.num_of_linear_cell = 256
        self.num_of_linear_cell = 512
        self.n_place_cells = 256
        self.n_head_cells = 12
        self.seq_len = 100
        self.model_input_size = 3
        self.batch_size = 24
        self.epochs = 100
        self.interval_of_save_weight = 3
        self.learning_rate = 1e-5
        self.momentum = 0.9
        self.is_resume = True
        self.clipping_max_norm = 20.
        self.clipping_norm_type = 2
        self.training_clipping = 1e-5
        self.weight_decay = 1e-5
Config = ConfigCLASS()
