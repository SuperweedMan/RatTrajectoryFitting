class ConfigCLASS:
    def __init__(self):
        self.add_noise = False
        self.noise_variance = 0.06
        self.data_root_path = '../GridCellDataset/npy'
        self.check_point_path = './checkpoint_v13'
        self.random_seed = 1
        # self.n_place_cells = 256
        self.dropout_rate = 0.5
        self.num_of_linear_cell = 256
        # self.num_of_linear_cell = 512
        self.n_place_cells = 256
        self.n_head_cells = 12
        self.seq_len = 100
        self.model_input_size = 3
        self.batch_size = 10
        self.epochs = 50
        self.interval_of_save_weight = 3
        self.learning_rate = 1e-5
        self.is_resume = True
        self.clipping_max_norm = 20.
        self.clipping_norm_type = 2
        self.weight_decay = 1e-5
Config = ConfigCLASS()
