def read_config(config_file):
    setup_config = {}

    setup_config["gpu"] = config_file["mode"]["gpu"]

    setup_config["tmp_files"] = setup_config["paths"]["tmp_data_path"]

    setup_config["source"] = setup_config["source"]["source"]
    setup_config["set"] = setup_config["source"]["set"]

    setup_config["num_epochs"] = setup_config["general"]["num_epochs"]
    setup_config["batch_size"] = setup_config["hyperparams"]["batch_size"]
    setup_config["lr"] = setup_config["hyperparams"]["lr"]

    