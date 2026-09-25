# 第 4 章　看透每一次对话：HTTP 日志钩子

## 4.1 为什么 Agent 调试离不开日志

调试普通程序，你对着堆栈找 bug；调试 Agent，你要对着**模型看到的和说出的每一个字**找问题：

- 模型为什么不调用工具？——可能它根本没看到工具定义
- 输出解析为什么失败？——看看模型到底输出了什么格式
- Observation 为什么没生效？——检查它有没有真的被拼进 messages

这些问题只有一种答案方式：**把每次 HTTP 请求和响应的原始内容完整打出来**。老练的 Agent 开发者会告诉你：日志占了调试工作量的八成。

## 4.2 httpx 事件钩子：无侵入的日志方案

`openai` SDK 底层用 `httpx` 发请求，而 `httpx` 提供事件钩子（event hooks）——在请求发出后、响应返回后自动执行我们指定的函数。这意味着**日志逻辑完全不用侵入业务代码**：

```python
import httpx

http_client = httpx.Client(
    event_hooks={
        "request": [log_request],
        "response": [log_response],
    }
)

client = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url=os.environ["ZHIPU_BASE_URL"],
    http_client=http_client,      # 把带钩子的客户端交给 openai SDK
)
```

`event_hooks` 是一个字典：`"request"` 键下挂请求发出后执行的函数列表，`"response"` 键下挂响应返回后执行的函数列表。以后无论业务代码怎么写，每一笔流量都自动被记录。

## 4.3 实现请求日志

```python
import json

def log_request(request: httpx.Request) -> None:
    print("\n" + "=" * 70)
    print(f">>> REQUEST  {request.method} {request.url}")
    print("Headers:", json.dumps(dict(request.headers),
                                 ensure_ascii=False, indent=2))
    body = request.content
    if body:
        try:
            print("Body:", json.dumps(json.loads(body.decode("utf-8")),
                                      ensure_ascii=False, indent=2))
        except Exception:
            print("Body(raw):", body.decode("utf-8", errors="replace"))
```

三个细节：

1. **`json.dumps(..., ensure_ascii=False, indent=2)`**——中文原样显示、按层级缩进。如果用默认的 `ensure_ascii=True`，你看到的将是 `\u4f60\u597d` 这样的转义串，什么都读不懂
2. **先 `json.loads` 再 `dumps`**——请求体是压缩的 JSON 字符串，重新格式化后才能一眼看清 messages 结构
3. **`try/except` 兜底**——万一请求体不是合法 JSON，退化为原始文本打印。**日志系统自身绝不能成为新的故障点**

## 4.4 响应日志与那个致命的 `.read()`

```python
def log_response(response: httpx.Response) -> None:
    # 关键：把 body 读出来并缓存，否则 hook 里拿不到
    response.read()
    print("\n" + "=" * 70)
    print(f"<<< RESPONSE  {response.status_code} {response.reason_phrase}")
    print("Headers:", json.dumps(dict(response.headers),
                                 ensure_ascii=False, indent=2))
    try:
        print("Body:", json.dumps(response.json(),
                                  ensure_ascii=False, indent=2))
    except Exception:
        print("Body(raw):", response.text)
```

整个文件里最容易踩的坑就是注释里的那行：

> **`response.read()` 必须调用，否则什么都拿不到。**

`httpx` 默认使用流式响应：钩子被调用时，响应头已到达，但**响应体还在网络上，没有被读入内存**。此时直接访问 `response.json()` 会失败或得到空。`response.read()` 把字节流读完并缓存进响应对象，之后的 `.json()` 才有数据可解析。

更妙的是：httpx 会**缓存已读内容**，openai SDK 随后再读同一个响应时用的是缓存，不会重复请求网络。一行代码，无副作用。

## 4.5 日志能回答哪些问题

装好钩子后，跑一次对话，终端会输出类似：

```
======================================================================
>>> REQUEST  POST https://open.bigmodel.cn/api/paas/v4/chat/completions
Headers: { "authorization": "Bearer xxx...", ... }
Body: {
  "model": "glm-4",
  "messages": [
    {"role": "system", "content": "你是一个使用 ReAct 模式……"},
    {"role": "user", "content": "帮我统计字数"}
  ],
  "tools": [ ... ]
}
======================================================================
<<< RESPONSE  200 OK
Body: {
  "choices": [{ "message": { "role": "assistant",
                             "content": "Thought: 我需要先看目录……" } }]
}
```

有了它，你可以逐项自查：

| 症状 | 在日志里查什么 |
|------|----------------|
| 模型不知道工具 | 请求 Body 里 `tools` 字段是否存在、描述是否完整 |
| 格式总是解析失败 | 响应 Body 里模型实际输出的原文 |
| 密钥报错 401 | Headers 里 authorization 是否正确 |
| Observation 丢失 | 下一轮请求的 messages 里有没有 tool 消息 |

## 4.6 进阶：把日志落盘

终端日志一滚就没，可以顺手写进文件：

```python
from datetime import datetime

LOG_FILE = Path("logs") / f"agent_{datetime.now():%Y%m%d_%H%M%S}.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log_request(request: httpx.Request) -> None:
    text = f"\n>>> REQUEST {request.method} {request.url}\n{request.content.decode('utf-8', errors='replace')}\n"
    print(text[:2000])                 # 终端截断显示
    LOG_FILE.open("a", encoding="utf-8").write(text)  # 文件完整保存
```

每次运行一个日志文件，事后可以完整复盘 Agent 的每一步决策。**可观测性不是锦上添花，是 Agent 工程的地基**。

## 动手任务

> 1. 给你的客户端装上两个钩子，发起一次调用，通读完整日志
> 2. 故意注释掉 `response.read()`，观察会发生什么
> 3. 思考题：为什么请求日志不需要类似 `read()` 的操作？（提示：请求体是发送方构造的，本来就在内存里）

下一章，我们给模型装上"手脚"——工具系统。
