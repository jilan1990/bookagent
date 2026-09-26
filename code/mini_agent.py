"""最简单的 AI Agent：ReAct 模式（Thought → Action → Observation 循环）"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for line in Path(".env").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

def log_request(request: httpx.Request) -> None:
    print("\n" + "=" * 70)
    print(f">>> REQUEST  {request.method} {request.url}")
    print("Headers:", json.dumps(dict(request.headers), ensure_ascii=False, indent=2))
    body = request.content
    if body:
        try:
            print("Body:", json.dumps(json.loads(body.decode("utf-8")), ensure_ascii=False, indent=2))
        except Exception:
            print("Body(raw):", body.decode("utf-8", errors="replace"))

def log_response(response: httpx.Response) -> None:
    # 关键：把 body 读出来并缓存，否则 hook 里拿不到
    response.read()
    print("\n" + "=" * 70)
    print(f"<<< RESPONSE  {response.status_code} {response.reason_phrase}")
    print("Headers:", json.dumps(dict(response.headers), ensure_ascii=False, indent=2))
    try:
        print("Body:", json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print("Body(raw):", response.text)

client = OpenAI(api_key=os.environ["ZHIPU_API_KEY"], base_url=os.environ["ZHIPU_BASE_URL"],
    http_client=httpx.Client(
        event_hooks={
            "request": [log_request],
            "response": [log_response],
        }
    ),)


WORKSPACE = Path(__file__).resolve().parent

def _safe_path(path):
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE)):
        raise PermissionError("只能访问工作区内的文件")
    return p

registry = {}  # 工具注册表：函数名 -> {func, description, parameters}

def tool(description, parameters=None):
    def decorator(func):
        registry[func.__name__] = {
            "func": func,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        }
        return func
    return decorator

@tool("读取工作区内指定文本文件的全部内容并返回。当需要查看文件内容、确认文件当前状态时调用。"
      "只能访问工作区内的文件，路径相对于工作区根目录。", {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "文件相对路径，如：notes/todo.txt",
        },
    },
    "required": ["path"],
})
def read_file(path):
    return _safe_path(path).read_text(encoding="utf-8", errors="replace")

@tool("把内容写入工作区内的文本文件（覆盖写入）。当需要创建新文件或整体替换文件内容时调用。"
      "父目录不存在时会自动创建；如需在原内容基础上修改，应先用 read_file 读取再拼接写入。", {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "文件相对路径，如：output/result.txt",
        },
        "content": {
            "type": "string",
            "description": "要写入的完整文本内容",
        },
    },
    "required": ["path", "content"],
})
def write_file(path, content):
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.name}（{len(content)} 字符）"

@tool("在子进程中执行一段 Python 代码并返回标准输出。当其他工具无法完成、需要运行代码逻辑"
      "（如数据处理、调用标准库）时调用。代码在工作区目录下运行，最长执行 20 秒；"
      "执行出错时会返回错误信息。", {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "完整的 Python 代码，可多行，如：print(sum(range(10)))。"
                           "需要看到的结果必须用 print() 输出",
        },
    },
    "required": ["code"],
})
def run_python(code):
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=20, cwd=WORKSPACE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        return f"执行出错（退出码 {proc.returncode}）:\n{proc.stderr.strip()}"
    return proc.stdout.strip() or "（无输出）"

@tool("列出工作区内某个目录下的文件和子目录（仅一层，不递归）。当需要查看目录结构时调用。", {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "目录相对路径，如：docs。不传时默认为工作区根目录",
        },
    },
    "required": [],
})
def list_dir(path="."):
    entries = [
        f"{'目录' if item.is_dir() else '文件'}: {item.name}"
        for item in sorted(_safe_path(path).iterdir())
    ]
    return "\n".join(entries) or "（空目录）"

BROWSER_MAX_STEPS = 15     # 浏览器任务最多执行的动作步数
BROWSER_TIMEOUT = 300      # 浏览器任务超时（秒）

@tool("用浏览器完成一个网页任务（打开网页、搜索、点击、填表、提取信息等）并返回最终结果。"
      "当需要访问互联网、查询实时信息或操作网页时调用。任务通常需要 1-3 分钟，"
      "一次只描述一个明确目标，复杂需求应拆成多次调用。", {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "要完成的网页任务，如：打开 baidu.com 搜索'北京今天天气'，返回当前温度",
        },
    },
    "required": ["task"],
})
def browser(task):
    try:  # 延迟导入：未安装时其他工具不受影响
        import inspect
        import asyncio
        from browser_use import Agent
    except ImportError:
        return ("错误: 未安装 browser-use，请先执行:\n"
                "pip install browser-use\n"
                "playwright install chromium")

    def _make_llm():  # 复用智谱 OpenAI 兼容接口；兼容新旧版 browser-use 的 LLM 封装
        kwargs = dict(base_url=os.environ["DEEPSEEK_BASE_URL"],
                      api_key=os.environ["DEEPSEEK_API_KEY"],
                      model=os.environ["DEEPSEEK_MODEL"])
        try:
            from langchain_openai import ChatOpenAI  # 旧版依赖
            kwargs["temperature"] = 0
        except ImportError:
            from browser_use import ChatOpenAI  # browser-use >= 0.3 自带
        return ChatOpenAI(**kwargs)

    def _browser_kwargs():  # 无头模式；兼容 0.3 / 0.2 的浏览器配置 API
        headless = os.environ.get("BROWSER_HEADLESS", "1") == "1"
        for attempt in ("new", "old"):
            try:
                if attempt == "new":
                    from browser_use import BrowserProfile, BrowserSession
                    return {"browser_session": BrowserSession(
                        browser_profile=BrowserProfile(headless=headless))}
                from browser_use import Browser, BrowserConfig
                return {"browser": Browser(config=BrowserConfig(headless=headless))}
            except ImportError:
                continue
        return {}  # 更旧的版本使用默认配置

    async def _run():
        extra = {"use_vision": False, **_browser_kwargs()}  # GLM 文本模型关闭视觉
        sig = inspect.signature(Agent)
        extra = {k: v for k, v in extra.items() if k in sig.parameters}
        agent = Agent(task=task, llm=_make_llm(), **extra)
        try:
            history = await agent.run()
            return history.final_result()
        finally:
            # 显式关闭浏览器，避免进程挂住
            if hasattr(agent, 'browser') and agent.browser:
                await agent.browser.close()

    try:
        result = asyncio.run(asyncio.wait_for(_run(), timeout=BROWSER_TIMEOUT))
        return str(result) if result else "浏览器任务未产生最终结果（可能未完成），请拆小任务后重试"
    except asyncio.TimeoutError:
        return f"错误: 浏览器任务超过 {BROWSER_TIMEOUT} 秒未完成，请拆小任务后重试"
    except Exception as exc:
        return f"浏览器任务执行失败: {exc}"

def tools_spec():
    """按智谱 function calling 的 tools 参数格式生成工具 schema 列表"""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": item["description"],
                "parameters": item["parameters"],
            },
        }
        for name, item in registry.items()
    ]

def tools_prompt():
    """把 tools_spec 渲染成 JSON 文本，注入 ReAct 系统提示词"""
    return json.dumps(tools_spec(), ensure_ascii=False, indent=2)

SYSTEM_PROMPT = f"""你是一个使用 ReAct 模式解决问题的智能体。严格按以下格式输出，不要输出格式之外的内容：

Thought: 简要思考当前状况和下一步行动
Action: 工具名（必须是工具列表中的名称）
Action Input: 调用参数，必须是合法的 JSON 对象，例如 {{"path": "docs"}}

## 工具列表（智谱 tools 格式，Action Input 即对应 parameters 里的字段）
{tools_prompt()}

## 约束
1. 每次只能调用一个工具，Action Input 必须是合法 JSON（无参数时传 {{}}）
2. 工具结果会以 Observation 返回给你，请基于 Observation 继续思考，不要编造工具结果
3. 当已有足够信息时，输出：
Thought: 已经得到答案
Final Answer: 最终答案（用中文）"""


def parse_react(text):
    """解析模型输出，返回 (最终答案, None) 或 (None, (工具名, 参数JSON))。

    Action / Final Answer 只在行首匹配，且 Action 的工具名必须真实存在，
    避免 Thought 等正文内容提到这些标记词时被误判、导致提前结束循环。
    """
    # 真正的工具调用：行首 Action + 注册表中的工具名 + 行首 Action Input
    action = re.search(r"^Action:\s*(\w+)", text, re.MULTILINE)
    if action and action.group(1) in registry:
        rest = text[action.end():]
        marker = re.search(r"^Action Input:", rest, re.MULTILINE)
        if marker:
            params = {}
            brace = rest.find("{", marker.end())
            if brace != -1:
                try:  # raw_decode 支持嵌套大括号，取到第一个完整 JSON 对象为止
                    params, _ = json.JSONDecoder().raw_decode(rest[brace:])
                except json.JSONDecodeError:
                    params = {}
            return None, (action.group(1), json.dumps(params, ensure_ascii=False))

    # Final Answer 只认行首标记，且取最后一次出现，把其后内容作为答案
    finals = list(re.finditer(r"^Final Answer:\s*", text, re.MULTILINE))
    if finals:
        return text[finals[-1].end():].strip(), None

    # 没有有效的 Action / Final Answer：交给 run_tool 报未知工具，促使模型纠正输出格式
    return None, (
        action.group(1) if action else None,
        "{}",
    )


def run_tool(action, action_input):
    item = registry.get(action)
    if not item:
        return f"错误: 未知工具 {action}，可用工具: {', '.join(registry)}"
    try:
        result = str(item["func"](**json.loads(action_input or "{}")))
    except Exception as exc:
        result = f"错误: {exc}"
    #if len(result) > 1500:
    #    result = result[:1500] + f"\n...（已截断，共 {len(result)} 字符）"
    return result


# ── 历史消息压缩 ──────────────────────────────────────────────
MAX_HISTORY_TOKENS = 8000   # 历史消息估算 token 上限，超过则压缩
KEEP_RECENT_MESSAGES = 6    # 压缩时保留最近几条消息（保证当前 ReAct 轮次完整）

# 压缩用的独立客户端（不挂日志 hook，避免刷屏）
summary_client = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"], base_url=os.environ["ZHIPU_BASE_URL"],
    timeout=httpx.Timeout(60.0, connect=10.0),
)

def estimate_tokens(text):
    """粗略估算中英混合文本的 token 数：中日韩字符约 1 token/字，其余约 4 字符/token"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(text) - cjk + 3) // 4

SUMMARY_PREFIX = "[此前对话摘要]"

def compress_history(messages):
    """历史消息超过上限时压缩旧消息。

    保护规则：系统提示词、当前任务目标永远原样保留；
    最后一条 assistant 消息之后的内容（如待 LLM 分析的 Observation）是未分析数据，永不压缩。
    """
    total = sum(estimate_tokens(m["content"]) for m in messages)
    if total <= MAX_HISTORY_TOKENS or len(messages) <= 2:
        return messages

    # 任务目标 = 最近一条真实用户输入（非 Observation/摘要消息），从它起属于当前任务
    task_start = 1
    for i, m in enumerate(messages):
        if m["role"] == "user" and not m["content"].startswith(("Observation:", SUMMARY_PREFIX)):
            task_start = i

    # 未分析数据边界：最后一条 assistant 消息之后的内容（如待分析的 Observation）不能压缩
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_idx:
        return messages
    tail_start = min(len(messages) - KEEP_RECENT_MESSAGES, assistant_idx[-1] + 1)
    tail_start = max(tail_start, task_start + 1)  # 任务目标在 tail 之前单独保留

    # 优先压缩当前任务之前的往期对话；往期无可压缩内容时，压缩当前任务中已分析过的中间步骤
    old, scope = messages[1:task_start], "往期对话"
    if len(old) < 2:
        old, scope = messages[task_start + 1:tail_start], "当前任务中间步骤"
    if len(old) < 2:  # 没有可安全压缩的内容
        return messages

    print(f"\n[压缩] 历史约 {total} tokens，正在压缩{scope} {len(old)} 条消息…")
    resp = summary_client.chat.completions.create(
        model=os.environ["ZHIPU_MODEL"],
        messages=[
            {"role": "system", "content":
                "把以下对话历史压缩成尽量简短的摘要，必须保留：用户的任务目标、已完成的步骤、"
                "关键工具结果和数据、尚未完成的事项。直接输出摘要正文。"},
            {"role": "user", "content": json.dumps(old, ensure_ascii=False)},
        ],
    )
    summary = {"role": "user", "content": f"{SUMMARY_PREFIX}\n{resp.choices[0].message.content.strip()}"}
    if scope == "往期对话":
        # 系统提示词 + 当前任务（含任务目标）整段原样保留
        compressed = [messages[0], summary] + messages[task_start:]
    else:
        # 系统提示词 + 任务目标原样保留，只替换已分析过的中间步骤；
        # tail_start 之后（最后一条 assistant 与待分析的 Observation）原样保留
        compressed = [messages[0], messages[task_start], summary] + messages[tail_start:]
    print(f"[压缩] 完成：{total} -> {sum(estimate_tokens(m['content']) for m in compressed)} tokens")
    return compressed


messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        user_input = input("\n你> ").strip()
    except EOFError:
        break
    if user_input.lower() in ("exit", "quit", "q"):
        break
    if not user_input:
        continue
    messages.append({"role": "user", "content": user_input})

    for step in range(1, 29):  # 最多 8 步 Thought → Action → Observation
        messages = compress_history(messages)  # 调用前先压缩，避免超出 context 上限
        text = client.chat.completions.create(
            model=os.environ["ZHIPU_MODEL"], messages=messages
        ).choices[0].message.content

        messages.append({"role": "assistant", "content": text})
        print(f"\n─── 第 {step} 轮 ───\n{text.strip()}")

        final, action_step = parse_react(text)
        if final:
            print(f"\nAgent> {final}")
            break

        action, action_input = action_step
        observation = run_tool(action, action_input)
        print(f"[Observation] {action}({action_input}) -> {observation}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    else:
        print("\nAgent> 已达到最大步数，任务中止。")
