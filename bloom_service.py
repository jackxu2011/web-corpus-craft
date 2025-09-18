import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rbloom import Bloom
from hashlib import sha256
from loguru import logger

# 全局Bloom Filter实例
bloom_filter: Optional[Bloom] = None

# 数据模型定义
class AddRequest(BaseModel):
    elements: List[str]

class CheckRequest(BaseModel):
    elements: List[str]

class CheckResponse(BaseModel):
    results: Dict[str, bool]

class StatusResponse(BaseModel):
    size_in_bits: int
    element_count: float
    error_rate: float
    capacity: int

# 哈希函数定义
def hash_func(obj):
    """自定义哈希函数，将字符串转换为整数哈希值"""
    if isinstance(obj, str):
        obj = obj.encode('utf-8')
    h = sha256(obj).digest()
    return int.from_bytes(h[:16], "big", signed=True)

# 服务生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化Bloom Filter，关闭时保存状态"""
    global bloom_filter
    args = app.state.args

    # 尝试从文件加载Bloom Filter
    if os.path.exists(args.save_path):
        try:
            bloom_filter = Bloom.load(args.save_path, hash_func)
            logger.info(f"从 {args.save_path} 加载Bloom Filter")
        except Exception as e:
            logger.info(f"加载Bloom Filter失败，创建新实例: {e}")
            bloom_filter = Bloom(args.capacity, args.error_rate, hash_func)
    else:
        bloom_filter = Bloom(args.capacity, args.error_rate, hash_func)
        logger.info(f"创建新Bloom Filter，容量: {args.capacity}, 误判率: {args.error_rate}")
        logger.info(bloom_filter.size_in_bits)
    # 启动定期保存任务
    save_interval = args.save_interval
    if save_interval > 0:
        async def periodic_save():
            while True:
                await asyncio.sleep(save_interval)
                if bloom_filter:
                    try:
                        bloom_filter.save(args.save_path)
                        logger.info(f"定期保存Bloom Filter到 {args.save_path}，元素数: {bloom_filter.approx_items}")
                    except Exception as e:
                        logger.info(f"保存Bloom Filter失败: {e}")
        asyncio.create_task(periodic_save())

    yield  # 服务运行期间

    # 服务关闭时保存
    if bloom_filter:
        try:
            bloom_filter.save(args.save_path)
            logger.info(f"服务关闭，保存Bloom Filter到 {args.save_path}")
        except Exception as e:
            logger.info(f"服务关闭时保存失败: {e}")

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan, title="Bloom Filter Service")

# 接口实现
@app.post("/add", response_model=Dict[str, str])
async def add_elements(request: AddRequest):
    """添加元素到Bloom Filter"""
    if bloom_filter is None:
        raise HTTPException(status_code=500, detail="Bloom Filter未初始化")

    if not request.elements:
        return {"message": "未提供元素"}

    # 批量添加元素
    try:
        bloom_filter.update(request.elements)
        return {"message": f"成功添加 {len(request.elements)} 个元素"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加元素失败: {str(e)}")

@app.post("/check", response_model=CheckResponse)
async def check_elements(request: CheckRequest):
    """检查元素是否存在于Bloom Filter中"""
    if bloom_filter is None:
        raise HTTPException(status_code=500, detail="Bloom Filter未初始化")

    if not request.elements:
        return CheckResponse(results={})

    # 批量检查元素
    results = {}
    for elem in request.elements:
        elem = str(elem)
        results[elem] = elem in bloom_filter

    return CheckResponse(results=results)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取Bloom Filter状态信息"""
    if bloom_filter is None:
        logger.info(f'Bloom Filter未初始化 ')
        raise HTTPException(status_code=500, detail="Bloom Filter未初始化")

    args = app.state.args

    return StatusResponse(
        size_in_bits=bloom_filter.size_in_bits,
        element_count=bloom_filter.approx_items,
        error_rate=args.error_rate,
        capacity=args.capacity
    )

@app.post("/clear", response_model=Dict[str, str])
async def clear_filter():
    """清空Bloom Filter（谨慎使用）"""
    global bloom_filter
    if bloom_filter is None:
        raise HTTPException(status_code=500, detail="Bloom Filter未初始化")

    args = app.state.args
    bloom_filter.clear()
    return {"message": "Bloom Filter已清空"}

# 启动服务
def main():
    parser = argparse.ArgumentParser(description="Bloom Filter 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--capacity", type=int, default=10_000_000, help="初始容量")
    parser.add_argument("--error-rate", type=float, default=0.001, help="允许的误判率")
    parser.add_argument("--save-path", type=str, default="bloom_filter.dat", help="Bloom Filter保存路径")
    parser.add_argument("--save-interval", type=int, default=300, help="定期保存间隔(秒)，0表示不自动保存")
    args = parser.parse_args()

    # 存储参数到应用状态
    app.state.args = args

    # 启动服务（使用uvicorn运行）
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
