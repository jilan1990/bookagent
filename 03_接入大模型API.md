# 第 3 章　第一个模型调用：接入大模型 API

## 3.1 OpenAI 兼容接口：事实标准

如今几乎所有大模型服务商都提供 **OpenAI 兼容接口**——请求格式、响应格式与 OpenAI 的 `/v1/chat/completions` 完全一致，只是 `base_url` 和模型名不同。

这意味着我们只需学一种 API 写法，就能对接市面上绝大多数模型。本书以智谱为例：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url=os.environ["ZHIPU_BASE_URL"],   # https://open.bigmodel.cn/api/paas/v4/
)
```

`openai` 这个 SDK 只是一个"壳"：它把 Python 对象翻译成 HTTP 请求发出去。换供应商 = 换 `base_url`，仅此而已。

## 3.2 消息结构：对话的最小单位

一切对话都由 `messages` 列表构成，每条消息有角色：

```python
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user",   "content": "你好，介绍一下你自己"},
]
```

| 角色 | 含义 | 在 Agent 中的用途 |
|------|------|------------------|
| `system` | 立法者：定身份、定规则 | 放 Agent 的人设、输出格式契约、工具说明 |
| `user` | 用户输入 | 放任务目标 |
| `assistant` | 模型的回复 | 放模型的思考与行动 |
| `tool` | 工具执行结果 | 放 Observation（部分协议中叫 `function`） |

**记住这张表**，第 6 章的 ReAct 循环本质上就是在不断往这个列表里追加 `assistant` 和 `tool` 消息。

## 3.3 发起第一次调用

```python
response = client.chat.completions.create(
    model=os.environ.get("ZHIPU_MODEL", "glm-4"),
    messages=messages,
    temperature=0.6,
)

print(response.choices[0].message.content)
```

几个参数值得抠一下：

- `model`：模型名。不同供应商命名不同，放进 `.env` 里配置
- `temperature`：随机性，0~1。**写 Agent 建议用 0~0.6**：我们要求模型稳定输出可解析的格式，随机性太高格式就飘
- 返回值的结构是 `choices[0].message.content`——一层层剥开才能拿到文本

## 3.4 流式调用：为什么本书不用它

SDK 支持 `stream=True` 逐字返回，聊天产品用它降低等待感。但本书的 Agent **不用流式**，原因有三：

1. **解析需要完整文本**：ReAct 的解析器要从整段输出里提取 Thought / Action，流式的碎片没有意义
2. **日志需要完整请求/响应**：第 4 章的钩子机制对流式响应要额外拼帧，复杂度陡增
3. **Agent 一轮要跑十几秒**，瓶颈在工具执行与多轮循环，首字快慢用户无感

工具型 Agent 求稳，聊天产品求快——场景决定技术选型。

## 3.5 把调用包成函数

为后续复用，我们把调用封装成一个小函数：

```python
def call_llm(client, messages):
    """调用大模型，返回回复文本"""
    response = client.chat.completions.create(
        model=os.environ.get("ZHIPU_MODEL", "glm-4"),
        messages=messages,
        temperature=0.6,
    )
    return response.choices[0].message.content
```

跑一下：

```python
answer = call_llm(client, [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "用一句话解释什么是Agent"},
])
print(answer)
```

至此，Agent 有了"大脑"。但它还只会说话——它甚至不知道自己说过什么，因为**每次调用都是独立的，对话历史要靠我们自己维护**（`messages` 列表就是它的记忆，第 9 章详谈）。

## 3.6 异常处理先行

网络会超时、密钥会过期、额度会用完。给调用加上最基本的防护：

```python
import time

def call_llm_safe(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return call_llm(client, messages)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt          # 1s, 2s, 4s 指数退避
            print(f"调用失败({e})，{wait}s 后重试...")
            time.sleep(wait)
```

指数退避（exponential backoff）是调用外部服务的标准姿势：失败后等 1 秒重试，再失败等 2 秒、4 秒……给服务端喘息时间，也给偶发故障自愈机会。

## 动手任务

> 1. 完成 `call_llm_safe`，故意写错 API Key，观察重试日志
> 2. 把 `temperature` 分别调成 0 和 1，问同一个问题五次，感受差异

下一章我们给这套调用装上"X 光机"——把每次 HTTP 请求和响应的原始内容都看清楚。
