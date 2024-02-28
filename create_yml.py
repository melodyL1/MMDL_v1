import yaml



def update_yaml_file(model, dataset, data, y,
                     started, end, train_ndvi, train_vv,
                     train_vh, train_band,train_percent ,validation_percent):
    file_path = "configs/config.yml"
    with open(file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # 更新配置文件中的参数值
    config['model'] = model
    config['data']['dataset'] = dataset
    config['data']['data'] = data
    config['data']['y'] = y
    config['data']['started'] = int(started)
    config['data']['end'] = int(end)
    config['data']['train_ndvi'] = int(train_ndvi)
    config['data']['train_vv'] = int(train_vv)
    config['data']['train_vh'] = int(train_vh)
    config['data']['train_band'] = int(train_band)
    config['data']['tr_percent'] = float(train_percent)
    config['data']['val_percent'] = float(validation_percent)
    #config['train']['optimizer']['betas'] = [ 0.8, 0.99 ]

    with open(file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)

