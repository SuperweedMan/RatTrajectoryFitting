class ConfigCLASS:
    def __init__(self):
        self.data_root_path = '../GridCellDataset/npy'
        self.check_point_path = './checkpoint_v6'
        self.random_seed = 1
        self.n_place_cells = 256
        self.n_head_cells = 12
        self.seq_len = 100
        self.model_input_size = 3
        self.batch_size = 1
        self.epochs = 100
        self.interval_of_save_weight = 3
        self.learning_rate = 1e-5
        self.is_resume = False
        self.clipping_max_norm = 20.
        self.clipping_norm_type = 2
Config = ConfigCLASS()
