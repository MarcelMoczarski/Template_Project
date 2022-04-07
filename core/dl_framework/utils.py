def read_config(config_file):
    setup_config = {}
    # abr_dict = {}
    for key, value in config_file.items():
        if key == "title":
            continue
        else:
            for subkey, subval in value.items():
                if type(subval) != dict:
                    key_name = f"{key[0]}_{subkey}"
                    setup_config[key_name] = subval
                else:
                    for subsubkey, subsubval in subval.items():
                        key_name = f"{key[0]}_{subkey[0]}_{subsubkey}"
                        setup_config[key_name] = subsubval
    return setup_config

