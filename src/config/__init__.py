HYPERPARAMETERS = {
    "simclr": [
        #{"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.05},  # "d": 512, 
        #{"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.07},  # "d": 512, 
        #{"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.1},   # "d": 512, 
        #{"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.2},   # "d": 512, 
        #{"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.3},   # "d": 512, 
        #{"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.4},   # "d": 512, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.05},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.07},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.1},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.2},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.3},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.4},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.05},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.07},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.1},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.2},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.3},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.4},   # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-7, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-5, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-4, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-3, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-2, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.2, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.3, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.4, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.5, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.6, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
        {"batch_size": 2048, "lr": 0.8, "weight_decay": 1e-6, "temperature" : 0.15},  # "d": 2048, 
    ],
    "vicreg": [
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.3},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.5},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.6},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.7},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.8},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.9},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 1},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 2},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 8},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 16},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 5, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 10, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 15, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 20, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 30, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 35, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 40, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 45, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.3, "weight_decay": 1e-06, "sim_coeff": 50, "cov_coeff": 25, "std_coeff": 4},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.3},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.4},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.5},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.6},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.7},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-07, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.8},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-06, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 0.9},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 1e-05, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 1},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 0.0001, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 2},
        {"batch_size": 1024, "lr": 0.5, "weight_decay": 0.001, "sim_coeff": 25, "cov_coeff": 25, "std_coeff": 4}
    ],
    "vicreg-ctr": [
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 1, "temperature": 0.05},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 1, "temperature": 0.07},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 1, "temperature": 0.1},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 1, "temperature": 0.2},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 1, "temperature": 0.3},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 1, "temperature": 0.4},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 0.1, "temperature": 0.1},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 0.5, "temperature": 0.1},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 2, "temperature": 0.1},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 4, "temperature": 0.1},
        {"batch_size": 1024, "learning_rate": 0.5, "weight_decay": 1e-6, "sim_coeff": 1, "cov_coeff": 1, "std_coeff": 8, "temperature": 0.1},
    ]
}