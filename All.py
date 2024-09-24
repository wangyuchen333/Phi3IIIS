import  yaml
import json

yaml_data = """
compute_environment: "LOCAL_MACHINE"
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: NO
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

# 将YAML数据转换为Python字典
data = yaml.safe_load(yaml_data)

# 将Python字典转换为JSON字符串
json_data = json.dumps(data, indent=4)

print(json_data)