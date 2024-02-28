import yaml



def update_yaml_file(data):
    file_path = "configs/config.yml"
    with open(file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # 更新配置文件中的参数值
    config['data']['data'] = data

    with open(file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)

