from fastapi import FastAPI, Request, HTTPException
import uvicorn, logging, json
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import os
from fastapi.responses import StreamingResponse
import litellm, uuid, time
from dotenv import load_dotenv
import sys

load_dotenv()

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
for l in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(l).setLevel(logging.WARNING)


class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]
        return (
            not any(p in str(record.msg) for p in blocked)
            if hasattr(record, "msg")
            else True
        )


logging.getLogger().addFilter(MessageFilter())

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.5")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

OPENAI_MODELS = {"gpt-4.5", "gpt-4o", "gpt-4o-mini", "gpt-4.1"}
GEMINI_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"}


def clean_gemini_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        if (
            schema.get("type") == "string"
            and "format" in schema
            and schema["format"] not in {"enum", "date-time"}
        ):
            schema.pop("format")
        for k, v in list(schema.items()):
            schema[k] = clean_gemini_schema(v)
    elif isinstance(schema, list):
        return [clean_gemini_schema(i) for i in schema]
    return schema


def map_model(v: str) -> tuple:
    original = v
    for p, l in [("anthropic/", 10), ("openai/", 7), ("gemini/", 7)]:
        if v.startswith(p):
            v = v[l:]
            break
    mapped = False
    new_model = v
    if PREFERRED_PROVIDER == "anthropic":
        new_model = f"anthropic/{v}"
        mapped = True
    elif "haiku" in v.lower():
        new_model = (
            f"gemini/{SMALL_MODEL}"
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS
            else f"openai/{SMALL_MODEL}"
        )
        mapped = True
    elif "sonnet" in v.lower():
        new_model = (
            f"gemini/{BIG_MODEL}"
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS
            else f"openai/{BIG_MODEL}"
        )
        mapped = True
    elif v in GEMINI_MODELS and not original.startswith("gemini/"):
        new_model = f"gemini/{v}"
        mapped = True
    elif v in OPENAI_MODELS and not original.startswith("openai/"):
        new_model = f"openai/{v}"
        mapped = True
    if not mapped and not original.startswith(("openai/", "gemini/", "anthropic/")):
        logger.warning(
            f"No prefix or mapping rule for model: '{original}'. Using as is."
        )
    return new_model, original


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict], Dict, List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_field(cls, v, info):
        new_model, original = map_model(v)
        if isinstance(info.data, dict):
            info.data["original_model"] = original
        return new_model


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator("model")
    def validate_model_token_count(cls, v, info):
        new_model, original = map_model(v)
        if isinstance(info.data, dict):
            info.data["original_model"] = original
        return new_model


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


def parse_tool_result_content(content):
    if content is None:
        return "No content provided"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(parse_tool_result_content(item) for item in content).strip()
    if isinstance(content, dict):
        return (
            content.get("text", "")
            if content.get("type") == "text"
            else json.dumps(content)
        )
    return str(content)


def convert_anthropic_to_litellm(req: MessagesRequest) -> Dict[str, Any]:
    messages = []
    if req.system:
        if isinstance(req.system, str):
            messages.append({"role": "system", "content": req.system})
        elif isinstance(req.system, list):
            system_text = "".join(
                b.text + "\n\n"
                for b in req.system
                if hasattr(b, "type")
                and b.type == "text"
                or isinstance(b, dict)
                and b.get("type") == "text"
            )
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    for msg in req.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        elif msg.role == "user" and any(
            getattr(b, "type", "") == "tool_result" for b in msg.content
        ):
            text_content = "".join(
                f"Tool result for {getattr(b, 'tool_use_id', '')}:\n{parse_tool_result_content(getattr(b, 'content', ''))}\n"
                for b in msg.content
                if getattr(b, "type", "") == "tool_result"
            )
            messages.append({"role": "user", "content": text_content.strip()})
        else:
            processed = []
            for block in msg.content:
                t = getattr(block, "type", "")
                if t == "text":
                    processed.append({"type": "text", "text": block.text})
                elif t == "image":
                    processed.append({"type": "image", "source": block.source})
                elif t == "tool_use":
                    processed.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif t == "tool_result":
                    c = getattr(block, "content", None)
                    content_list = (
                        [{"type": "text", "text": c}]
                        if isinstance(c, str)
                        else (
                            c
                            if isinstance(c, list)
                            else [{"type": "text", "text": str(c)}]
                        )
                    )
                    processed.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": getattr(block, "tool_use_id", ""),
                            "content": content_list,
                        }
                    )
            messages.append({"role": msg.role, "content": processed})

    max_tokens = req.max_tokens
    if req.model.startswith("openai/") or req.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)

    litellm_req = {
        "model": req.model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }
    if req.thinking and req.model.startswith("anthropic/"):
        litellm_req["thinking"] = req.thinking
    if req.stop_sequences:
        litellm_req["stop"] = req.stop_sequences
    if req.top_p:
        litellm_req["top_p"] = req.top_p
    if req.top_k:
        litellm_req["top_k"] = req.top_k

    if req.tools:
        is_gemini = req.model.startswith("gemini/")
        openai_tools = []
        for tool in req.tools:
            tool_dict = tool.dict() if hasattr(tool, "dict") else tool
            input_schema = (
                clean_gemini_schema(tool_dict.get("input_schema", {}))
                if is_gemini
                else tool_dict.get("input_schema", {})
            )
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_dict["name"],
                        "description": tool_dict.get("description", ""),
                        "parameters": input_schema,
                    },
                }
            )
        litellm_req["tools"] = openai_tools

    if req.tool_choice:
        tc = (
            req.tool_choice.dict()
            if hasattr(req.tool_choice, "dict")
            else req.tool_choice
        )
        t = tc.get("type")
        if t == "auto":
            litellm_req["tool_choice"] = "auto"
        elif t == "any":
            litellm_req["tool_choice"] = "any"
        elif t == "tool" and "name" in tc:
            litellm_req["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }
        else:
            litellm_req["tool_choice"] = "auto"

    return litellm_req


def convert_litellm_to_anthropic(
    lr: Union[Dict, Any], original: MessagesRequest
) -> MessagesResponse:
    try:
        clean_model = original.model
        for p, l in [("anthropic/", 10), ("openai/", 7)]:
            if clean_model.startswith(p):
                clean_model = clean_model[l]
                break
        is_claude = clean_model.startswith("claude-")

        if hasattr(lr, "choices") and hasattr(lr, "usage"):
            choice = lr.choices[0] if lr.choices else None
            message = choice.message if choice else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            finish_reason = choice.finish_reason if choice else "stop"
            usage_info = lr.usage
            response_id = getattr(lr, "id", f"msg_{uuid.uuid4()}")
        else:
            rd = (
                lr
                if isinstance(lr, dict)
                else (
                    lr.dict()
                    if hasattr(lr, "dict")
                    else (lr.model_dump() if hasattr(lr, "model_dump") else lr.__dict__)
                )
            )
            choice = rd.get("choices", [{}])[0]
            message = choice.get("message", {})
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls")
            finish_reason = choice.get("finish_reason", "stop")
            usage_info = rd.get("usage", {})
            response_id = rd.get("id", f"msg_{uuid.uuid4()}")

        content = []
        if content_text:
            content.append({"type": "text", "text": content_text})

        if tool_calls and is_claude:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tc in tool_calls:
                func = (
                    tc.get("function", {})
                    if isinstance(tc, dict)
                    else getattr(tc, "function", None)
                )
                tid = (
                    tc.get("id", f"tool_{uuid.uuid4()}")
                    if isinstance(tc, dict)
                    else getattr(tc, "id", f"tool_{uuid.uuid4()}")
                )
                name = (
                    func.get("name", "")
                    if isinstance(func, dict)
                    else getattr(func, "name", "")
                    if func
                    else ""
                )
                args = (
                    func.get("arguments", "{}")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", "{}")
                    if func
                    else "{}"
                )
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {"raw": args}
                content.append(
                    {"type": "tool_use", "id": tid, "name": name, "input": args}
                )
        elif tool_calls and not is_claude:
            tool_text = "\n\nTool usage:\n"
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            for tc in tool_calls:
                func = (
                    tc.get("function", {})
                    if isinstance(tc, dict)
                    else getattr(tc, "function", None)
                )
                name = (
                    func.get("name", "")
                    if isinstance(func, dict)
                    else getattr(func, "name", "")
                    if func
                    else ""
                )
                args = (
                    func.get("arguments", "{}")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", "{}")
                    if func
                    else "{}"
                )
                args_str = json.dumps(
                    json.loads(args) if isinstance(args, str) else args, indent=2
                )
                tool_text += f"Tool: {name}\nArguments: {args_str}\n\n"
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})

        prompt_tokens = (
            usage_info.get("prompt_tokens", 0)
            if isinstance(usage_info, dict)
            else getattr(usage_info, "prompt_tokens", 0)
        )
        completion_tokens = (
            usage_info.get("completion_tokens", 0)
            if isinstance(usage_info, dict)
            else getattr(usage_info, "completion_tokens", 0)
        )

        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }.get(finish_reason, "end_turn")
        if not content:
            content.append({"type": "text", "text": ""})

        return MessagesResponse(
            id=response_id,
            model=original.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )
    except Exception as e:
        import traceback

        logger.error(f"Error converting response: {str(e)}\n{traceback.format_exc()}")
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


async def handle_streaming(gen, original: MessagesRequest):
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        tool_index = None
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        async for chunk in gen:
            try:
                if hasattr(chunk, "usage") and chunk.usage:
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0)

                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    delta = (
                        choice.delta
                        if hasattr(choice, "delta")
                        else getattr(choice, "message", {})
                    )
                    finish_reason = getattr(choice, "finish_reason", None)

                    delta_content = getattr(delta, "content", None) or (
                        delta.get("content") if isinstance(delta, dict) else None
                    )

                    if delta_content:
                        accumulated_text += delta_content
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    delta_tool_calls = getattr(delta, "tool_calls", None) or (
                        delta.get("tool_calls") if isinstance(delta, dict) else None
                    )

                    if delta_tool_calls:
                        if tool_index is None:
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            elif (
                                accumulated_text
                                and not text_sent
                                and not text_block_closed
                            ):
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tc in delta_tool_calls:
                            idx = (
                                tc.get("index")
                                if isinstance(tc, dict)
                                else getattr(tc, "index", 0)
                            )
                            if tool_index is None or idx != tool_index:
                                tool_index = idx
                                last_tool_index += 1
                                func = (
                                    tc.get("function", {})
                                    if isinstance(tc, dict)
                                    else getattr(tc, "function", None)
                                )
                                name = (
                                    func.get("name", "")
                                    if isinstance(func, dict)
                                    else getattr(func, "name", "")
                                    if func
                                    else ""
                                )
                                tid = (
                                    tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                                    if isinstance(tc, dict)
                                    else getattr(
                                        tc, "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                    )
                                )
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': last_tool_index, 'content_block': {'type': 'tool_use', 'id': tid, 'name': name, 'input': {}}})}\n\n"

                            args = (
                                tc.get("function", {}).get("arguments", "")
                                if isinstance(tc, dict)
                                else ""
                            ) or (
                                getattr(getattr(tc, "function", None), "arguments", "")
                                if hasattr(tc, "function")
                                else ""
                            )
                            if args:
                                try:
                                    args_json = (
                                        json.dumps(args)
                                        if isinstance(args, dict)
                                        else (
                                            args if json.loads(args) or True else args
                                        )
                                    )
                                except:
                                    args_json = str(args)
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': last_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        sr = {
                            "length": "max_tokens",
                            "tool_calls": "tool_use",
                            "stop": "end_turn",
                        }.get(finish_reason, "end_turn")
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': sr, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        if not has_sent_stop_reason:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"
    except Exception as e:
        import traceback

        logger.error(f"Error in streaming: {str(e)}\n{traceback.format_exc()}")
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")
        display_model = (
            original_model.split("/")[-1] if "/" in original_model else original_model
        )

        litellm_req = convert_anthropic_to_litellm(request)

        if request.model.startswith("openai/"):
            litellm_req["api_key"] = OPENAI_API_KEY
            if OPENAI_BASE_URL:
                litellm_req["api_base"] = OPENAI_BASE_URL
        elif request.model.startswith("gemini/"):
            if USE_VERTEX_AUTH:
                litellm_req["vertex_project"] = VERTEX_PROJECT
                litellm_req["vertex_location"] = VERTEX_LOCATION
                litellm_req["custom_llm_provider"] = "vertex_ai"
            else:
                litellm_req["api_key"] = GEMINI_API_KEY
        else:
            litellm_req["api_key"] = ANTHROPIC_API_KEY

        if "openai" in litellm_req["model"]:
            for i, msg in enumerate(litellm_req["messages"]):
                if "content" in msg and isinstance(msg["content"], list):
                    text_content = ""
                    for block in msg["content"]:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_content += block.get("text", "") + "\n"
                            elif block.get("type") == "tool_result":
                                rc = block.get("content", [])
                                if isinstance(rc, list):
                                    for item in rc:
                                        if isinstance(item, dict):
                                            text_content += (
                                                item.get("text", json.dumps(item))
                                                + "\n"
                                            )
                                elif isinstance(rc, str):
                                    text_content += rc + "\n"
                                else:
                                    text_content += json.dumps(rc) + "\n"
                            elif block.get("type") == "tool_use":
                                text_content += f"[Tool: {block.get('name', 'unknown')} (ID: {block.get('id', 'unknown')})]\nInput: {json.dumps(block.get('input', {}))}\n\n"
                            elif block.get("type") == "image":
                                text_content += "[Image content]\n"
                    litellm_req["messages"][i]["content"] = (
                        text_content.strip() or "..."
                    )
                elif msg.get("content") is None:
                    litellm_req["messages"][i]["content"] = "..."
                for key in list(msg.keys()):
                    if key not in [
                        "role",
                        "content",
                        "name",
                        "tool_call_id",
                        "tool_calls",
                    ]:
                        del msg[key]

        if request.stream:
            response_gen = await litellm.acompletion(**litellm_req)
            return StreamingResponse(
                handle_streaming(response_gen, request), media_type="text/event-stream"
            )
        else:
            litellm_resp = litellm.completion(**litellm_req)
            anthropic_resp = convert_litellm_to_anthropic(litellm_resp, request)
            return anthropic_resp
    except Exception as e:
        import traceback

        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        for attr in ["message", "status_code", "response", "llm_provider", "model"]:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        logger.error(f"Error: {json.dumps(error_details, indent=2)}")
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        from litellm import token_counter

        converted = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            )
        )
        args = {"model": converted["model"], "messages": converted["messages"]}
        if request.model.startswith("openai/") and OPENAI_BASE_URL:
            args["api_base"] = OPENAI_BASE_URL
        token_count = token_counter(**args)
        return TokenCountResponse(input_tokens=token_count)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return TokenCountResponse(input_tokens=1000)


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
