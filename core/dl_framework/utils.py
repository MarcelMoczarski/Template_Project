def read_config(config_file):
    setup_config = {}

    setup_config["gpu"] = config_file["mode"]["gpu"]

    setup_config["tmp_files"] = config_file["paths"]["tmp_data_path"]

    setup_config["num_epochs"] = config_file["general"]["num_epochs"]
    setup_config["valid_split"] = config_file["general"]["valid_split"]
    
    setup_config["source"] = config_file["source"]["source"]
    setup_config["set"] = config_file["source"]["set"]

    setup_config["batch_size"] = config_file["hyperparams"]["batch_size"]
    setup_config["lr"] = config_file["hyperparams"]["lr"]

    return setup_config
    