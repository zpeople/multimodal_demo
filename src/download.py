import os
import argparse
import requests
from huggingface_hub import snapshot_download
# source /etc/network_turbo
# unset http_proxy && unset https_proxy
# python download.py --model "Qwen/Qwen3-4B-Instruct-2507" --output /root/autodl-tmp/model


def check_mirror_availability(mirror_url):
    """检查镜像源是否可用"""
    try:
        # 尝试访问镜像源的基础URL
        response = requests.head(mirror_url, timeout=10)
        return response.status_code < 400
    except Exception as e:
        print(f"镜像源 {mirror_url} 不可用: {str(e)}")
        return False

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='从国内镜像源下载Hugging Face模型')
    parser.add_argument('--model', required=True, help='要下载的模型名称（Hugging Face仓库ID）')
    parser.add_argument('--output', help='模型保存目录（默认：当前目录下的model文件夹）')
    parser.add_argument('--mirror', choices=['aliyun', 'hf_mirror'], default='aliyun', 
                      help='选择镜像源（aliyun或hf_mirror，默认：aliyun）')
    parser.add_argument('--force', action='store_true', help='强制重新下载，即使本地目录已存在')
    
    args = parser.parse_args()
    
    # 镜像源URL映射
    mirror_urls = {
        'aliyun': 'https://mirrors.aliyun.com/huggingface/',
        'hf_mirror': 'https://hf-mirror.com'
        
    }
    
    # 设置并验证镜像源
    selected_mirror = mirror_urls[args.mirror]
    os.environ["HF_ENDPOINT"] = selected_mirror
    print(f"已设置镜像源: {selected_mirror}")
    
    # 检查镜像源可用性
    if not check_mirror_availability(selected_mirror):
        print("镜像源不可用，尝试切换到另一个镜像源...")
        # 切换到另一个镜像源
        other_mirror = 'hf_mirror' if args.mirror == 'aliyun' else 'aliyun'
        selected_mirror = mirror_urls[other_mirror]
        os.environ["HF_ENDPOINT"] = selected_mirror
        print(f"已切换到镜像源: {selected_mirror}")
        
        if not check_mirror_availability(selected_mirror):
            print("所有镜像源都不可用，请检查网络连接")
            return
    
    # 确定保存路径
    if args.output:
        local_model_dir = os.path.join(args.output, args.model)
    else:
        local_model_dir = os.path.join(os.getcwd(), "model", args.model)
    
    # 如果强制重新下载，删除已存在的目录
    if args.force and os.path.exists(local_model_dir):
        import shutil
        print(f"强制删除已存在的目录: {local_model_dir}")
        shutil.rmtree(local_model_dir)
    
    # 创建保存目录
    os.makedirs(local_model_dir, exist_ok=True)  
    
    print(f"开始从{args.mirror}镜像下载模型：{args.model}，保存路径：{local_model_dir}")
    
    try:
        # 下载模型
        snapshot_download(
            repo_id=args.model,              
            local_dir=local_model_dir,    
            local_dir_use_symlinks=False,  
            allow_patterns=["*.json", "*.bin", "*.txt", "*.model", "*.png", "*.safetensors"],  
            resume_download=True,          
            ignore_patterns=["*.git*", "*.md", "*.h5"],  
            max_workers=4                  
        )
        print(f"模型 {args.model} 下载完成！")
    except Exception as e:
        print(f"下载过程中出错: {str(e)}")
        print("建议尝试：")
        print("1. 检查网络连接")
        print("2. 使用--force参数强制重新下载")
        print("3. 尝试另一个镜像源 (--mirror hf_mirror)")

if __name__ == "__main__":
    main()
    