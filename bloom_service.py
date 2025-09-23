import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Tuple, Any
import hashlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rbloom import Bloom
from loguru import logger

# 全局Bloom Filter分片实例列表
bloom_filters: List[Optional[Bloom]] = []
# 分片数量
NUM_SHARDS = 1000

# 数据模型定义
class AddRequest(BaseModel):
    elements: List[str]

class CheckRequest(BaseModel):
    elements: List[str]

class CheckResponse(BaseModel):
    results: Dict[str, bool]

class StatusResponse(BaseModel):
    total_size_in_bits: int
    total_element_count: float
    error_rate: float
    shard_capacity: int
    shard_count: int

# 哈希分片工具函数
def get_shard_index(element: str) -> int:
    """计算元素应归属的分片索引（0到NUM_SHARDS-1）"""
    hash_obj = hashlib.sha256(element.encode('utf-8'))
    hash_val = int.from_bytes(hash_obj.digest()[:8], "big")  # 使用前8字节计算哈希值
    return hash_val % NUM_SHARDS

def element_hash_func(obj):
    """自定义元素哈希函数，将字符串转换为整数哈希值"""
    if isinstance(obj, str):
        obj = obj.encode('utf-8')
    h = hashlib.sha256(obj).digest()
    return int.from_bytes(h[:16], "big", signed=True)

# 分片管理函数
def get_shard_path(base_path: str, shard_idx: int) -> str:
    """获取指定分片的存储路径"""
    dir_name = os.path.dirname(base_path)
    file_name = os.path.basename(base_path)
    name, ext = os.path.splitext(file_name)
    return os.path.join(dir_name, f"{name}_shard_{shard_idx}{ext}")

async def init_shards(args) -> List[Bloom]:
    """初始化所有Bloom过滤器分片"""
    shards = []
    for i in range(NUM_SHARDS):
        shard_path = get_shard_path(args.save_path, i)
        try:
            if os.path.exists(shard_path):
                shard = Bloom.load(shard_path, element_hash_func)
                logger.info(f"从 {shard_path} 加载分片 {i}")
            else:
                shard = Bloom(args.capacity, args.error_rate, element_hash_func)
                logger.info(f"创建新分片 {i}，容量: {args.capacity}, 误判率: {args.error_rate}")
            shards.append(shard)
        except Exception as e:
            logger.error(f"初始化分片 {i} 失败: {e}")
            # 尝试创建新的分片
            shards.append(Bloom(args.capacity, args.error_rate, element_hash_func))
    return shards

async def save_shards(shards: List[Bloom], base_path: str) -> Tuple[int, int]:
    """保存所有分片，返回成功和失败的数量"""
    success = 0
    failed = 0
    for i, shard in enumerate(shards):
        if shard is None:
            failed += 1
            continue
        try:
            shard_path = get_shard_path(base_path, i)
            shard.save(shard_path)
            success += 1
        except Exception as e:
            logger.error(f"保存分片 {i} 失败: {e}")
            failed += 1
    return success, failed

# 服务生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化Bloom Filter分片，关闭时保存状态"""
    global bloom_filters
    args = app.state.args

    # 确保存储目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 初始化所有分片
    bloom_filters = await init_shards(args)
    logger.info(f"完成 {len(bloom_filters)} 个Bloom Filter分片的初始化")

    # 启动定期保存任务
    save_interval = args.save_interval
    if save_interval > 0:
        async def periodic_save():
            while True:
                await asyncio.sleep(save_interval)
                if bloom_filters:
                    success, failed = await save_shards(bloom_filters, args.save_path)
                    total_elements = sum(shard.approx_items for shard in bloom_filters if shard)
                    logger.info(
                        f"定期保存完成: 成功 {success} 个, 失败 {failed} 个, "
                        f"总元素数: {total_elements}"
                    )
        asyncio.create_task(periodic_save())

    yield  # 服务运行期间

    # 服务关闭时保存所有分片
    if bloom_filters:
        success, failed = await save_shards(bloom_filters, args.save_path)
        logger.info(
            f"服务关闭，保存完成: 成功 {success} 个, 失败 {failed} 个"
        )

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan, title="Distributed Bloom Filter Service")

# 接口实现
@app.post("/add", response_model=Dict[str, str])
async def add_elements(request: AddRequest):
    """添加元素到对应的Bloom Filter分片"""
    if not bloom_filters or any(shard is None for shard in bloom_filters):
        raise HTTPException(status_code=500, detail="Bloom Filter分片未完全初始化")

    if not request.elements:
        return {"message": "未提供元素"}

    # 统计各分片添加的元素数量
    shard_counts = [0] * NUM_SHARDS

    # 批量添加元素到对应的分片
    try:
        for elem in request.elements:
            shard_idx = get_shard_index(elem)
            bloom_filters[shard_idx].add(elem)
            shard_counts[shard_idx] += 1

        # 找出有数据的分片
        active_shards = sum(1 for count in shard_counts if count > 0)
        return {
            "message": f"成功添加 {len(request.elements)} 个元素，分布在 {active_shards} 个分片中"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加元素失败: {str(e)}")

@app.post("/check", response_model=CheckResponse)
async def check_elements(request: CheckRequest):
    """检查元素是否存在于对应的Bloom Filter分片中"""
    if not bloom_filters or any(shard is None for shard in bloom_filters):
        raise HTTPException(status_code=500, detail="Bloom Filter分片未完全初始化")

    if not request.elements:
        return CheckResponse(results={})

    # 批量检查元素
    results = {}
    for elem in request.elements:
        shard_idx = get_shard_index(elem)
        results[elem] = elem in bloom_filters[shard_idx]

    return CheckResponse(results=results)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取所有Bloom Filter分片的状态信息"""
    if not bloom_filters or any(shard is None for shard in bloom_filters):
        logger.error("Bloom Filter分片未完全初始化")
        raise HTTPException(status_code=500, detail="Bloom Filter分片未完全初始化")

    args = app.state.args

    # 计算总状态
    total_size = sum(shard.size_in_bits for shard in bloom_filters)
    total_elements = sum(shard.approx_items for shard in bloom_filters)

    return StatusResponse(
        total_size_in_bits=total_size,
        total_element_count=total_elements,
        error_rate=args.error_rate,
        shard_capacity=args.capacity,
        shard_count=NUM_SHARDS
    )

@app.get("/shard-status/{shard_idx}", response_model=Dict[str, Any])
async def get_shard_status(shard_idx: int):
    """获取指定分片的状态信息"""
    if shard_idx < 0 or shard_idx >= NUM_SHARDS:
        raise HTTPException(status_code=400, detail=f"分片索引必须在0到{NUM_SHARDS-1}之间")

    if not bloom_filters or bloom_filters[shard_idx] is None:
        raise HTTPException(status_code=500, detail=f"Bloom Filter分片 {shard_idx} 未初始化")

    shard = bloom_filters[shard_idx]
    return {
        "shard_idx": shard_idx,
        "size_in_bits": shard.size_in_bits,
        "element_count": shard.approx_items,
        "error_rate": shard.error_rate,
        "capacity": shard.capacity
    }

@app.post("/clear", response_model=Dict[str, str])
async def clear_filter():
    """清空所有Bloom Filter分片（谨慎使用）"""
    global bloom_filters
    if not bloom_filters or any(shard is None for shard in bloom_filters):
        raise HTTPException(status_code=500, detail="Bloom Filter分片未完全初始化")

    args = app.state.args
    for shard in bloom_filters:
        shard.clear()

    return {"message": f"所有 {NUM_SHARDS} 个Bloom Filter分片已清空"}

@app.post("/clear-shard/{shard_idx}", response_model=Dict[str, str])
async def clear_shard(shard_idx: int):
    """清空指定的Bloom Filter分片（谨慎使用）"""
    if shard_idx < 0 or shard_idx >= NUM_SHARDS:
        raise HTTPException(status_code=400, detail=f"分片索引必须在0到{NUM_SHARDS-1}之间")

    if not bloom_filters or bloom_filters[shard_idx] is None:
        raise HTTPException(status_code=500, detail=f"Bloom Filter分片 {shard_idx} 未初始化")

    bloom_filters[shard_idx].clear()
    return {"message": f"Bloom Filter分片 {shard_idx} 已清空"}

# 启动服务
def main():
    parser = argparse.ArgumentParser(description="分布式Bloom Filter服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--capacity", type=int, default=5_000_000, help="每个分片的初始容量")
    parser.add_argument("--error_rate", type=float, default=0.001, help="允许的误判率")
    parser.add_argument("--save_path", type=str, default="tmp/bloom_filter.dat", help="Bloom Filter分片的基础保存路径")
    parser.add_argument("--save_interval", type=int, default=0, help="定期保存间隔(秒)，0表示不自动保存")
    args = parser.parse_args()

    # 存储参数到应用状态
    app.state.args = args

    # 启动服务（使用uvicorn运行）
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
