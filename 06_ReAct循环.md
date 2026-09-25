# 第6章 ReAct循环：Agent的心脏

前五章我们把零件备齐了：第 3 章接上了大脑（模型调用），第 4 章装好了神经（日志钩子），第 5 章长出了手脚（工具系统）。但零件堆在一起不会自己动——还需要一个心脏，把"思考、行动、观察"泵成一个循环。这就是 ReAct 循环，整个 harness 的灵魂所在。有趣的是，它的核心逻辑只有几十行；真正的工作量，全在围绕它的"契约"与"容错"上。

## 6.1 ReAct：推理与行动的交替

ReAct（Reasoning + Acting）源自 2022 年的同名论文，核心思想只有一句话：**让模型交替输出推理（Thought）与行动（Action），并把行动的真实结果（Observation）喂回给它**。

一轮循环长这样：

```
Thought: 我需要先看看工作区里有什么文件
Action: list_dir
Action Input: {}
（工具执行，返回 Observation）

Observation: 文件: mini_agent.py
目录: book

Thought: 根目录有 mini_agent.py，先读它
Action: read_file
Action Input: {"path": "mini_agent.py"}
...
```

与纯思维链（只让模型"想"）相比，ReAct 的关键优势是**推理接地**：模型不必靠想象编造事实，每一步都踩在真实工具返回的地面上。与纯 function calling（只让模型"做"）相比，Thought 把决策过程显式写了出来——既提升准确率，也让人类随时能看懂 agent 正在干什么。第 4 章的日志钩子负责"看得见"，Thought 负责"看得懂"。

## 6.2 契约：SYSTEM_PROMPT

agent 没有任何隐藏逻辑，所有游戏规则都写在系统提示词里。它本质上是**我们与模型签订的一纸契约**：

```python
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
```

四个设计点：

- **输出格式给模板**：不描述格式，而是直接给出样例。模型对"照着填空"的遵循度远高于"按如下规范"；
- **工具清单动态注入**：`{tools_prompt()}` 把第 5 章生成的 schema 文本填进提示词。新增工具时契约自动更新，这正是第 5 章"单一事实来源"设计的回报；
- **`{{}}` 转义**：f-string 里想输出字面的 `{"path": "docs"}`，必须写成 `{{"path": "docs"}}`。这是 f-string 的经典陷阱，漏写一个花括号，整个提示词在运行时直接抛异常；
- **约束写死**：每轮只调一个工具、不得编造 Observation、结束用 Final Answer。第 2 条尤其重要——模型天然有"讨好"倾向，不明确禁止，它会在没调工具时替你想象一个结果。

> 这种把工具清单写进提示词、用文本标记（`Observation:`）传递结果的方式，称为**文本协议**。它不依赖任何 API 特性，任何能聊天的模型都能跑。第 10 章会讲如何迁移到原生 function calling 协议。

## 6.3 状态：messages 就是 agent 的全部记忆

agent 没有数据库，没有状态机，全部状态就是一个列表：

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
```

每一轮循环做三次追加：

```python
messages.append({"role": "assistant", "content": text})          # 模型刚说的话
messages.append({"role": "user", "content": f"Observation: {observation}"})  # 工具结果
```

然后整个列表原样再发给 API。这里有个不得不做的妥协：chat API 只认 system / user / assistant 三种角色，没有"工具结果"这个角色，所以我们把 Observation 伪装成 user 消息，靠 `"Observation: "` 前缀做标记。这个前缀一物两用：对模型，它是"这不是真人说话"的信号；对程序，第 9 章的 `compress_history` 靠它区分真实用户输入和工具回填——**一处标记，两处受益**。

## 6.4 parse_react：解析的艺术

模型输出是自由文本，harness 要从中可靠地抠出 Action 和参数。这是最不起眼、却最决定成败的一段代码：

```python
def parse_react(text):
    """解析模型输出，返回 (最终答案, None) 或 (None, (工具名, 参数JSON))。"""
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

    # Final Answer 只认行首标记，且取最后一次出现
    finals = list(re.finditer(r"^Final Answer:\s*", text, re.MULTILINE))
    if finals:
        return text[finals[-1].end():].strip(), None

    # 没有有效的 Action / Final Answer：交给 run_tool 报未知工具，促使模型纠正格式
    return None, (
        action.group(1) if action else None,
        "{}",
    )
```

逐个细节拆解：

1. **`re.MULTILINE` + 行首 `^`**：只认行首的标记。如果匹配写得宽松，模型在 Thought 里写"我考虑用 Action: read_file"这样一句分析，就会被误判成真的调用；
2. **`action.group(1) in registry`**：工具名必须真实存在。模型会"幻觉"出不存在的工具名，这层校验拦住它们；
3. **`rest = text[action.end():]`**：Action Input 必须出现在 Action 之后，顺序不可颠倒；
4. **`raw_decode` 而非 `json.loads`**：这是整段代码最精妙的一处。参数 JSON 可能嵌套对象、数组，正则很难正确找到它的结束位置；`raw_decode` 从 `{` 开始解析，取到**第一个完整 JSON 对象**就停，天然支持任意嵌套；
5. **Final Answer 取最后一次**：模型有时会自我反思、重复格式，最后一个才是它真正的结论；
6. **兜底分支**：格式坏掉时构造 `(None, "{}")`，让 `run_tool` 报"未知工具 None"。这个错误会作为 Observation 回到模型面前，模型看到后通常一轮就修正了输出格式。**用反馈回路修格式，比硬报错退出优雅得多**。

## 6.5 run_tool：统一的异常边界

```python
def run_tool(action, action_input):
    item = registry.get(action)
    if not item:
        return f"错误: 未知工具 {action}，可用工具: {', '.join(registry)}"
    try:
        result = str(item["func"](**json.loads(action_input or "{}")))
    except Exception as exc:
        result = f"错误: {exc}"
    return result
```

短小，但有几个刻意的决定：

- **未知工具不抛异常**，而是返回带可用工具清单的错误文本——模型看到清单，下一步就会用对名字；
- **所有异常一律捕获**，转成 `"错误: ..."` 字符串。`PermissionError`、`TimeoutExpired`、参数缺失的 `TypeError`……在这里殊途同归。错误不是终局，而是 Observation；
- **`str(...)` 兜底**：工具返回值哪怕不是字符串，也能安全拼进消息；
- 源码里还留着一段被注释掉的**输出截断**逻辑：工具返回太长会撑爆上下文。第 9 章的压缩机制是一种解法，截断是另一种——留给你按场景取舍。

## 6.6 主循环：for-else 与步数上限

心脏本身只有二十行：

```python
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

    for step in range(1, 29):
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
```

三个要点：

- **for-else**：Python 的小众语法，`else` 在 for 循环**未被 break** 时执行——恰好就是"跑满步数还没出答案"的分支，语义严丝合缝；
- **步数上限必须存在**：模型可能陷入同一工具的循环调用，没有上限的 agent 会无限烧钱。这里允许 28 轮，足够大多数任务；
- **`compress_history(messages)` 每轮调用**：长任务的历史会持续膨胀，这是上下文保险丝，第 9 章详解。

> 顺手指出源码里的一个小瑕疵：循环上方注释写着"最多 8 步"，代码却是 28 步——注释腐烂了。这提醒我们：**与代码不同步的注释比没有注释更糟**，改动语义时永远记得连注释一起改。

## 6.7 本章小结

| 组件 | 职责 |
| --- | --- |
| `SYSTEM_PROMPT` | 与模型的格式契约，动态注入工具清单 |
| `messages` | 唯一状态，Observation 以 user 角色回填 |
| `parse_react` | 文本协议解析，行首锚定 + raw_decode + 兜底纠错 |
| `run_tool` | 统一异常边界，一切错误皆 Observation |
| 主循环 | for-else 控制步数上限，串起整个流程 |

一轮循环的数据流：

```
用户输入 ──► messages.append(user)
                │
                ▼
   ┌───────── ReAct 循环（最多 28 轮）─────────┐
   │  compress_history ──► LLM ──► text       │
   │  parse_react ──► Final Answer？──是──► 输出│
   │        │ 否                               │
   │        ▼                                  │
   │  run_tool ──► Observation ──► 回填 user    │
   └───────────────────────────────────────────┘
```

动手任务：

1. 把 `range(1, 29)` 改成 `range(1, 4)`，交给 agent 一个复杂任务，观察"已达到最大步数"分支的行为，以及此时 messages 里已经积累了什么。
2. 在 `parse_react` 的兜底分支里加一行 `print("[格式告警] 未匹配到有效 Action/Final Answer")`，连续运行十个任务，统计格式告警率——你会对"契约的脆弱性"有直观认识。
3. 删掉 `action.group(1) in registry` 这层校验，然后诱导模型调用一个不存在的工具，观察两种写法下 agent 的恢复路径有何不同。
