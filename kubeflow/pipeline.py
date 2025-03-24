from kfp import compiler
from kfp import components
from kfp import dsl
from kfp.dsl import component
from kfp import kubernetes
from kfp import Client

SECRET_WANDB = "wandb-api-key"

@dsl.container_component
def model_train():
    return dsl.ContainerSpec(image='deepvolition/nano-transformer:latest', 
                             command=['/bin/sh'], 
                             args=['-c' ,' python3 main.py'])


@dsl.pipeline(name='simple-train-pipeline')
def pipeline():
    train_task = model_train()
    train_task.set_gpu_limit(1)
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=SECRET_WANDB,
        secret_key_to_env={'password': 'WANDB_API_KEY'})

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path='pipeline.yaml')
    
    Client().create_run_from_pipeline_func(pipeline, experiment_name='kubeflow-demo')