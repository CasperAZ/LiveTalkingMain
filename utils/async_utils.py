###############################################################################
#  异步辅助工具
###############################################################################

import asyncio
from typing import AsyncIterator, TypeVar

T = TypeVar("T")


async def merge_async_iters(*iterators: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    合并多个异步迭代器，按先到先得顺序 yield。
    所有迭代器结束后退出。
    """
    # 这个工具适合把多个异步来源“汇流”成一个输出流。
    # 在实时系统里很常见，比如多个事件源、多个消息源合并消费。
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    active = len(iterators)

    async def _drain(it: AsyncIterator[T]):
        nonlocal active
        try:
            async for item in it:
                await queue.put(item)
        finally:
            active -= 1
            if active == 0:
                await queue.put(sentinel)

    tasks = [asyncio.create_task(_drain(it)) for it in iterators]
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
    finally:
        for task in tasks:
            task.cancel()


async def async_queue_iter(q: asyncio.Queue, sentinel=None) -> AsyncIterator:
    """将 asyncio.Queue 转为异步迭代器，收到 sentinel 时退出"""
    # 有些逻辑天然适合写成 async for，
    # 这个函数就是把“基于队列”的接口转成“基于迭代”的接口。
    while True:
        item = await q.get()
        if item is sentinel:
            break
        yield item
