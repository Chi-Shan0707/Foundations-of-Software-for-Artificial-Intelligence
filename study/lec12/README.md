# L12: 智能体（Agent）

> 智能体核心概念、构成要素、CLI 智能体实现原理及代码实践。

---

## 目录

1. [智能体的核心定义](#1-智能体的核心定义)
2. [智能体的三大构成要素](#2-智能体的三大构成要素)
3. [智能体与大模型的关系](#3-智能体与大模型的关系)
4. [CLI 智能体的设计与实现](#4-cli-智能体的设计与实现)
5. [代码详解：l12-cli-agent.py](#5-代码详解l12-cli-agentpy)
6. [安全考虑与改进方向](#6-安全考虑与改进方向)
7. [课后作业](#7-课后作业)

---

## 1. 智能体的核心定义

### 1.1 什么是智能体？

智能体（Agent）本质上是一个**软件客户端**，它依赖大模型（服务端）进行推理和规划。

**核心定位**：
- 智能体 = 客户端（Client）
- 大模型 = 服务端（Server）
- 类比：浏览器（客户端）↔ Web 服务器（服务端）

### 1.2 智能体的双重角色

| 角色 | 说明 |
|------|------|
| **大模型的客户端** | 向大模型发送请求，获取推理结果 |
| **面向用户的服务提供方** | 接收用户请求，调用大模型，返回最终结果 |

---

## 2. 智能体的三大构成要素

### 2.1 工具（Tools）

**定义**：智能体需要工具来完成具体任务。

**工具类型**：

| 类型 | 示例 | 说明 |
|------|------|------|
| **Web 应用** | 计算器 API、搜索引擎 | 通过 HTTP 请求调用远程服务 |
| **技能（Skills）** | 软件使用手册、命令指南 | 封装复杂操作为简单接口 |
| **系统命令** | ls、cat、grep 等 | 直接执行操作系统命令 |

**关键点**：
- 工具集合要覆盖常见需求
- 技能文档需清晰（如命令格式）
- 目标是降低使用门槛，像把专业命令封装成"一键操作"

**实际例子**：
```
用户需求："查看当前目录下的文件"
智能体调用：ls 命令
返回结果：文件列表
```

### 2.2 记忆（Memory）

**定义**：智能体能记住历史对话和上下文，以便在后续交互中提供更准确的回答。

**记忆的重要性**：
- 现在的会话（session）通常有长度限制
- 未来更强大的工具可能会记录所有聊天历史，并按需查询

**实际应用**：
```
Coding Agent（编程助手）需要记住：
- 整个代码仓库的结构
- 之前的交互记录
- 项目的配置文件
```

**技术实现**：
- 建立自己的数据库存储所有对话内容
- 在需要时快速调用和重组信息
- 支持长上下文的查询和检索

### 2.3 规划（Planning）

**定义**：智能体能够根据用户请求，制定执行计划并调用相应的工具。

**规划流程**：
```
用户请求 → 理解意图 → 制定计划 → 选择工具 → 执行 → 返回结果
```

---

## 3. 智能体与大模型的关系

### 3.1 服务端 vs 客户端

| 维度 | 大模型（服务端） | 智能体（客户端） |
|------|------------------|------------------|
| **核心功能** | 推理、生成 | 调用、执行 |
| **运行位置** | 云端服务器 | 用户设备 |
| **交互方式** | API 调用 | 命令行/GUI |
| **存储内容** | 模型参数 | 工具、记忆、配置 |

### 3.2 上节课回顾：推理（Inference）

上节课讨论的 **inference（推理）** 属于服务端功能，而智能体则扮演着双重角色：
- 既是**大模型的客户端**
- 又是**面向用户的服务提供方**

### 3.3 用户需求的差异性

不同用户对模型的需求存在差异：
- 某些功能可能对部分用户有用
- 其他用户可能不需要这些功能
- 不同用户的需求不同，不同模型的能力也不同

---

## 4. CLI 智能体的设计与实现

### 4.1 目标

实现一个能理解并执行 Linux 命令行操作的智能体。

### 4.2 工作流程

```
用户通过自然语言提出请求（如"查看当前文件"）
         ↓
智能体将请求转化为具体的 Linux 命令（如 ls）
         ↓
智能体执行该命令并获取结果
         ↓
将执行结果和原始请求再次输入给大模型
         ↓
大模型进行总结并最终返回给用户
```

### 4.3 技术要点

1. **API 配置**：需要配置 API 密钥，以便连接和调用大模型
2. **Action 解析**：通过解析模型返回的 action 字段（如 `run_shell`）来判断是否需要执行外部命令
3. **安全考虑**：由于执行命令存在风险（如 `rm -rf`），建议增加用户二次确认的环节

### 4.4 Prompt 设计

```python
system_prompt = """
你是一个CLI智能体。
如果用户请求涉及系统操作（如ls, dir, cat, pwd等），请返回JSON格式：

{
  "action": "run_shell",
  "command": "实际要执行的命令"
}

否则正常回答。
"""
```

**关键点**：
- 明确定义何时需要执行命令
- 规范输出格式为 JSON
- 区分普通回答和工具调用

---

## 5. 代码详解：l12-cli-agent.py

### 5.1 代码结构

```
l12-cli-agent.py
├── OpenAI 客户端配置
├── 系统提示词（System Prompt）
├── run_shell() - 执行 shell 命令
├── strip_markdown() - 清理 markdown 格式
├── call_model() - 调用大模型
├── chat() - 主对话逻辑
└── main() - 主程序入口
```

### 5.2 核心函数解析

#### 5.2.1 OpenAI 客户端配置

```python
client = OpenAI(
    api_key=os.getenv('MODELSCOPE_API_KEY'),
    base_url='https://api-inference.modelscope.cn/v1'
)
MODEL = "ZhipuAI/GLM-5.1"
```

**说明**：
- 使用 ModelScope 的 API 接口
- 采用 OpenAI 兼容格式
- 模型选择智谱 AI 的 GLM-5.1

#### 5.2.2 执行 Shell 命令

```python
def run_shell(cmd: str) -> str:
    """执行shell命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)
```

**关键点**：
- 使用 `subprocess.run()` 执行命令
- `shell=True` 允许执行 shell 命令
- `capture_output=True` 捕获标准输出和错误输出
- 返回 stdout 和 stderr 的组合

#### 5.2.3 清理 Markdown 格式

```python
def strip_markdown(text: str) -> str:
    return re.sub(r'^```(?:json)?\s*', '', re.sub(r'\s*```\s*$', '', text)).strip()
```

**作用**：去除模型返回内容中的 markdown 代码块标记，便于 JSON 解析。

#### 5.2.4 调用大模型

```python
def call_model(msgs):
    response = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=0.2,
        max_tokens=512,
        stream=True,
    )
    reasoning = ""
    content = ""
    for chunk in response:
        if chunk.choices:
            rc = getattr(chunk.choices[0].delta, 'reasoning_content', '') or ''
            ac = chunk.choices[0].delta.content or ''
            if rc:
                reasoning += rc
            if ac:
                content += ac
    return content.strip()
```

**特点**：
- 使用流式输出（`stream=True`）
- 支持推理内容（`reasoning_content`）和正文内容（`content`）分离
- 低温度（`temperature=0.2`）确保输出稳定

#### 5.2.5 主对话逻辑

```python
def chat(user_input: str):
    messages.append({"role": "user", "content": user_input})
    content = call_model(messages)

    # 尝试解析是否是工具调用
    try:
        data = json.loads(strip_markdown(content))

        if isinstance(data, dict) and data.get("action") == "run_shell":
            cmd = data["command"]
            print(f"\n[EXECUTING] {cmd}\n")
            output = run_shell(cmd)
            print(f"[OUTPUT]\n{output}")

            # 把执行结果回传给模型
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"命令执行结果：\n{output}\n请总结结果。"
            })

            reply = call_model(messages)
        else:
            reply = content

    except Exception:
        # 不是 JSON 就当普通回答
        reply = content

    messages.append({"role": "assistant", "content": reply})
    return reply
```

**流程解析**：
1. 将用户输入添加到消息列表
2. 调用大模型获取响应
3. 尝试解析响应是否为 JSON 格式的工具调用
4. 如果是 `run_shell` 操作，执行命令并将结果回传给模型
5. 如果不是工具调用，直接返回模型响应
6. 将最终回复添加到消息列表

### 5.3 程序运行示例

```
=== CLI Agent with Execution ===

You: 查看当前目录下的文件

[EXECUTING] ls

[OUTPUT]
README.md
l12-cli-agent.py
notes/

Agent: 当前目录下有以下文件和文件夹：
- README.md（文档文件）
- l12-cli-agent.py（Python 脚本）
- notes/（文件夹）

You: 查看 README.md 的内容

[EXECUTING] cat README.md

[OUTPUT]
（文件内容）

Agent: README.md 文件包含以下内容...
```

---

## 6. 安全考虑与改进方向

### 6.1 安全风险

- **危险命令**：`rm -rf`、`chmod 777` 等可能造成不可逆损害
- **权限问题**：执行命令可能需要特定权限
- **注入攻击**：恶意用户可能构造特殊输入

### 6.2 改进建议

| 改进方向 | 具体措施 |
|----------|----------|
| **二次确认** | 对危险命令（如删除、修改权限）增加用户确认环节 |
| **命令白名单** | 只允许执行预定义的安全命令 |
| **沙箱环境** | 在容器或虚拟机中执行命令，限制影响范围 |
| **日志记录** | 记录所有执行的命令和结果，便于审计 |
| **超时控制** | 对长时间运行的命令设置超时 |
| **权限控制** | 使用最小权限原则，避免使用 root 权限 |

### 6.3 功能扩展

- **多工具支持**：集成更多工具（如文件编辑、网络请求）
- **记忆功能**：实现长期记忆，记住之前的交互
- **多模态支持**：处理图片、音频等非文本输入
- **并行执行**：同时执行多个独立命令
- **错误处理**：更智能的错误恢复机制

---

## 7. 课后作业

1. **运行代码**：下载并运行 `code/l12-cli-agent.py`，亲身体验 CLI 智能体的功能
2. **配置 API**：在阿里云上配置 API 密钥，并尝试修改代码中的 prompt，观察对智能体行为的影响
3. **功能扩展**：尝试为智能体添加新的工具支持（如文件编辑、网络搜索）

---

## 附录：代码文件索引

| 文件 | 说明 |
|------|------|
| `code/l12-cli-agent.py` | CLI 智能体完整实现 |
| `study/lec12/README.md` | 本笔记文档 |
| `notes/L12-智能体.pdf` | 课程讲义原文 |

---

> **运行命令备忘**：
> ```bash
> # 安装依赖
> pip install openai python-dotenv
>
> # 设置 API 密钥（在 .env 文件中）
> MODELSCOPE_API_KEY=your_api_key_here
>
> # 运行 CLI 智能体
> python code/l12-cli-agent.py
> ```
