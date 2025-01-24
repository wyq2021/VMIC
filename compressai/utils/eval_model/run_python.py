import os
import subprocess  # 用于运行外部命令
import datetime    # 用于标记时间戳

# 定义输出目录和命令参数
output_dir = "/HDD/wyq/wyq_data/full_examples/output/lms_color"
script_command = "python -m compressai.utils.eval_model"
model_paths = [
    "/HDD/wyq/tic_copy/pretrained/tic/1/checkpoint_best_loss.pth.tar",
    "/HDD/wyq/tic_copy/pretrained/tic/2/checkpoint_best_loss.pth.tar",
    "/HDD/wyq/tic_copy/pretrained/tic/3/SwinV2_checkpoint_best_loss.pth.tar",
    "/HDD/wyq/tic_copy/pretrained/tic/5/checkpoint_best_loss.pth.tar",
    "/HDD/wyq/tic_copy/pretrained/tic/7/checkpoint_best_loss.pth.tar"
    # 添加更多模型路径
]
output_file = "/HDD/wyq/wyq_data/tic_copy/experiment_results.txt"
checkpoint = "checkpoint"
architecture = "tic"
use_cuda = True  # 是否使用 CUDA

# 确保输出文件目录存在
output_dirname = os.path.dirname(output_file)
os.makedirs(output_dirname, exist_ok=True)

# 创建并写入结果文件头部信息
with open(output_file, "w") as f:
    f.write(f"Experiment Results - {datetime.datetime.now()}\n")
    f.write("="*50 + "\n")

# 遍历每个模型路径并运行命令
for model_path in model_paths:
    try:
        print(f"Running experiment for model: {model_path}")

        # 构建完整命令
        command = f"{script_command} {checkpoint} {output_dir} -a {architecture} -p {model_path}"
        if use_cuda:
            command += " --cuda"

        # 调用命令并捕获输出
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # 将结果写入文件
        with open(output_file, "a") as f:
            f.write(f"Model Path: {model_path}\n")
            f.write("Output:\n")
            f.write(result.stdout + "\n")
            f.write("="*50 + "\n")

        print(f"Finished model: {model_path}, results saved.\n")
    except Exception as e:
        print(f"Error occurred while processing {model_path}: {e}")
        with open(output_file, "a") as f:
            f.write(f"Error for model: {model_path}\n")
            f.write(f"Details: {str(e)}\n")
            f.write("="*50 + "\n")
