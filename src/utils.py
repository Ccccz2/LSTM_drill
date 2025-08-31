import os
from datetime import datetime

def get_save_dir(base_dir, is_error=False):
    """创建带时间戳的子文件夹路径"""
    time_str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    dir_name = f"{time_str}_error" if is_error else time_str
    save_dir = os.path.join(base_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_error_log(error_dir, e):
    """保存错误日志"""
    error_log = os.path.join(error_dir, "error_log.txt")
    with open(error_log, "w") as f:
        f.write(f"错误时间: {datetime.now()}\n")
        f.write(f"错误类型: {type(e).__name__}\n")
        f.write(f"错误详情:\n{str(e)}\n")
    return error_log