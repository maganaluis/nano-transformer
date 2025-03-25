from typing import Dict
from kfp import compiler
from kfp import dsl
from kfp import kubernetes

import kfp_client

SECRET_WANDB = "wandb-api-key"

@dsl.component( base_image='deepvolition/nano-transformer:latest')
def json_to_yaml(conf: Dict) -> str:
    import yaml
    return yaml.dump(conf, sort_keys=False)

@dsl.container_component
def model_train(conf: str):
    return dsl.ContainerSpec(image='deepvolition/nano-transformer:latest', 
                             command=['/bin/sh'], 
                             args=['-c' ,' python3 main.py', '--config_str', conf])

@dsl.pipeline(name='simple-train-pipeline')
def pipeline(conf: Dict= {}):
    json_to_yaml_task = json_to_yaml(conf=conf)
    train_task = model_train(conf=json_to_yaml_task.output)
    train_task.set_accelerator_type('nvidia.com/gpu')
    train_task.set_accelerator_limit(1)
    train_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=SECRET_WANDB,
        secret_key_to_env={'password': 'WANDB_API_KEY'})

if __name__ == '__main__':

    # initialize a KFPClientManager
    kfp_client = kfp_client.KFPClientManager(
        api_url="http://localhost:8080/pipeline",
        skip_tls_verify=True,

        dex_username="user@example.com",
        dex_password="12341234",

        # can be 'ldap' or 'local' depending on your Dex configuration
        dex_auth_type="local",
    )

    # get a newly authenticated KFP client
    # TIP: long-lived sessions might need to get a new client when their session expires
    kfp_client = kfp_client.create_kfp_client()

    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path='pipeline.yaml')
    
    # kfp_client.create_run_from_pipeline_func(pipeline, experiment_name='kubeflow-demo', namespace='kubeflow-user-example-com')