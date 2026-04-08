# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import asyncio
import dataclasses
import httpx
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import tiktoken
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI

from ...utils.prompt_utils import generate_mcp_system_prompt
from ..base_client import BaseClient

logger = logging.getLogger("tvir_agent")

import logging


# logger = logging.getLogger("tvir_agent")
# openai_logger = logging.getLogger("openai")
# openai_logger.setLevel(logging.DEBUG)  # 提升 OpenAI SDK 的日志级别

# httpx_logger = logging.getLogger("httpx")
# httpx_logger.setLevel(logging.DEBUG)  # 同时看 httpx 的日志


@dataclasses.dataclass
class OpenAIClient(BaseClient):
    @staticmethod
    def _serialize_tool_call(tool_call: Any) -> Dict[str, Any]:
        """Convert SDK tool call objects into plain dicts for message history."""
        function = getattr(tool_call, "function", None)
        return {
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(function, "name", None) if function else None,
                "arguments": (
                    getattr(function, "arguments", None) if function else None
                ),
            },
        }

    def _build_stream_response(
        self,
        content: str,
        finish_reason: str,
        usage: Any,
        tool_calls: List[Any],
    ):
        """Build a response-like object from streaming chunks."""

        class StreamResponse:
            def __init__(self, content, finish_reason, usage, tool_calls):
                self.choices = [
                    type(
                        "obj",
                        (object,),
                        {
                            "message": type(
                                "obj",
                                (object,),
                                {
                                    "role": "assistant",
                                    "content": content,
                                    "tool_calls": tool_calls,
                                },
                            )(),
                            "finish_reason": finish_reason,
                        },
                    )()
                ]
                self.usage = usage

        return StreamResponse(content, finish_reason, usage, tool_calls)

    @staticmethod
    def _merge_stream_tool_calls(
        aggregated_tool_calls: Dict[int, Dict[str, Any]],
        delta_tool_calls: Any,
    ) -> Dict[int, Dict[str, Any]]:
        """Accumulate partial tool call deltas from streaming responses."""
        if not delta_tool_calls:
            return aggregated_tool_calls

        for delta_tool_call in delta_tool_calls:
            index = getattr(delta_tool_call, "index", None)
            if index is None:
                index = len(aggregated_tool_calls)

            current = aggregated_tool_calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )

            if getattr(delta_tool_call, "id", None):
                current["id"] = delta_tool_call.id
            if getattr(delta_tool_call, "type", None):
                current["type"] = delta_tool_call.type

            function = getattr(delta_tool_call, "function", None)
            if function:
                if getattr(function, "name", None):
                    current["function"]["name"] += function.name
                if getattr(function, "arguments", None):
                    current["function"]["arguments"] += function.arguments

        return aggregated_tool_calls

    @staticmethod
    def _materialize_stream_tool_calls(
        aggregated_tool_calls: Dict[int, Dict[str, Any]]
    ) -> List[Any]:
        """Convert aggregated tool-call chunks into SDK-like objects."""
        tool_calls = []
        for _, tool_call in sorted(aggregated_tool_calls.items()):
            tool_calls.append(
                SimpleNamespace(
                    id=tool_call["id"],
                    type=tool_call["type"],
                    function=SimpleNamespace(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    ),
                )
            )
        return tool_calls

    def _create_client(self) -> Union[AsyncOpenAI, OpenAI]:
        """Create LLM client"""
        http_client_args = {"headers": {"x-upstream-session-id": self.task_id}}
        if self.async_client:
            return AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=DefaultAsyncHttpxClient(**http_client_args),
            )
        else:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=DefaultHttpxClient(**http_client_args),
            )

    def _update_token_usage(self, usage_data: Any) -> None:
        """Update cumulative token usage"""
        if usage_data:
            input_tokens = getattr(usage_data, "prompt_tokens", 0)
            output_tokens = getattr(usage_data, "completion_tokens", 0)
            prompt_tokens_details = getattr(usage_data, "prompt_tokens_details", None)
            if prompt_tokens_details:
                cached_tokens = (
                    getattr(prompt_tokens_details, "cached_tokens", None) or 0
                )
            else:
                cached_tokens = 0

            # Record token usage for the most recent call
            self.last_call_tokens = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }

            # OpenAI does not provide cache_creation_input_tokens
            self.token_usage["total_input_tokens"] += input_tokens
            self.token_usage["total_output_tokens"] += output_tokens
            self.token_usage["total_cache_read_input_tokens"] += cached_tokens

            self.task_log.log_step(
                "info",
                "LLM | Token Usage",
                f"Input: {self.token_usage['total_input_tokens']}, "
                f"Output: {self.token_usage['total_output_tokens']}",
            )

    async def _handle_stream_response(self, stream):
        """处理流式响应"""
        full_content = ""
        finish_reason = None
        usage_data = None
        aggregated_tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # 获取增量内容
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    content = choice.delta.content
                    if content:
                        full_content += content
                        # 这里可以添加回调函数来实时输出内容
                        # if self.stream_callback:
                        #     await self.stream_callback(content)

                # 获取结束原因
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason

                # 获取 token 使用情况（通常在最后一个 chunk）
                if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls"):
                    aggregated_tool_calls = self._merge_stream_tool_calls(
                        aggregated_tool_calls, choice.delta.tool_calls
                    )

                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage

            # 更新 token 使用统计
            if usage_data:
                self._update_token_usage(usage_data)

            # 构造类似非流式响应的对象
            class StreamResponse:
                def __init__(self, content, finish_reason, usage, tool_calls):
                    self.choices = [
                        type(
                            "obj",
                            (object,),
                            {
                                "message": type(
                                    "obj",
                                    (object,),
                                    {
                                        "role": "assistant",
                                        "content": content,
                                        "tool_calls": tool_calls,
                                    },
                                )(),
                                "finish_reason": finish_reason,
                            },
                        )()
                    ]
                    self.usage = usage

            response = StreamResponse(
                full_content,
                finish_reason,
                usage_data,
                self._materialize_stream_tool_calls(aggregated_tool_calls),
            )

            self.task_log.log_step(
                "info",
                "LLM | Response Status",
                f"{finish_reason or 'N/A'}",
            )

            return response

        except Exception as e:
            # self.task_log.log_step(
            #     "error",
            #     "LLM | Stream Error",
            #     f"Error processing stream: {str(e)}",
            # )
            raise e

    async def _create_message(
        self,
        system_prompt: str,
        messages_history: List[Dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
        stream: bool = False,
    ):
        """
        Send message to OpenAI API.
        :param system_prompt: System prompt string.
        :param messages_history: Message history list.
        :param stream: Whether to use streaming response.
        :return: OpenAI API response object or None (if error occurs).
        """

        # Create a copy for sending to LLM (to avoid modifying the original)
        messages_for_llm = [m.copy() for m in messages_history]

        # put the system prompt in the first message since OpenAI API does not support system prompt in
        if system_prompt:
            # Check if there's already a system or developer message
            if messages_for_llm and messages_for_llm[0]["role"] in [
                "system",
                "developer",
            ]:
                messages_for_llm[0] = {
                    "role": "system",
                    "content": system_prompt,
                }

            else:
                messages_for_llm.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

        # Filter tool results to save tokens (only affects messages sent to LLM)
        messages_for_llm = self._remove_tool_result_from_messages(
            messages_for_llm, keep_tool_result
        )

        # Retry loop with dynamic max_tokens adjustment
        max_retries = 10
        base_wait_time = 30
        current_max_tokens = self.max_tokens

        for attempt in range(max_retries):
            formatted_tools = []
            if tools_definitions and self.use_tool_calls is not False:
                formatted_tools = await self.convert_tool_definition_to_tool_call(
                    tools_definitions
                )

            params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": messages_for_llm,
                "tools": formatted_tools,
                "stream": stream,
                "top_p": self.top_p,
                "extra_body": {},
            }
            if formatted_tools:
                params["tool_choice"] = "auto"
            # Check if the model is GPT-5, and adjust the parameter accordingly
            if "gpt-5" in self.model_name:
                # Use 'max_completion_tokens' for GPT-5
                params["max_completion_tokens"] = current_max_tokens
            else:
                # Use 'max_tokens' for GPT-4 and other models
                params["max_tokens"] = current_max_tokens

            if "glm" in self.model_name:
                params["extra_body"] = {"thinking": {"type": "enabled"}}

            # Add repetition_penalty if it's not the default value
            if self.repetition_penalty != 1.0:
                params["extra_body"]["repetition_penalty"] = self.repetition_penalty

            if "deepseek-v3-1" in self.model_name:
                params["extra_body"]["thinking"] = {"type": "enabled"}

            try:
                self.task_log.log_step(
                    "info",
                    "LLM | Call Start",
                    f"Calling LLM ({'async' if self.async_client else 'sync'}) attempt {attempt + 1}/{max_retries}",
                )

                if self.async_client:
                    if stream:
                        # 流式响应
                        stream_response = await self.client.chat.completions.create(
                            **params
                        )
                        response = await self._handle_stream_response(stream_response)
                    else:
                        # 非流式响应
                        response = await self.client.chat.completions.create(**params)
                        self._update_token_usage(getattr(response, "usage", None))
                else:
                    if stream:
                        # 同步流式响应
                        stream_response = self.client.chat.completions.create(**params)
                        # 注意：同步版本需要不同的处理方式
                        full_content = ""
                        finish_reason = None
                        usage_data = None
                        aggregated_tool_calls: Dict[int, Dict[str, Any]] = {}

                        for chunk in stream_response:
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            if hasattr(choice, "delta") and hasattr(
                                choice.delta, "content"
                            ):
                                content = choice.delta.content
                                if content:
                                    full_content += content
                            if hasattr(choice, "delta") and hasattr(
                                choice.delta, "tool_calls"
                            ):
                                aggregated_tool_calls = (
                                    self._merge_stream_tool_calls(
                                        aggregated_tool_calls, choice.delta.tool_calls
                                    )
                                )
                            if (
                                hasattr(choice, "finish_reason")
                                and choice.finish_reason
                            ):
                                finish_reason = choice.finish_reason
                            if hasattr(chunk, "usage") and chunk.usage:
                                usage_data = chunk.usage

                        if usage_data:
                            self._update_token_usage(usage_data)

                        class StreamResponse:
                            def __init__(self, content, finish_reason, usage, tool_calls):
                                self.choices = [
                                    type(
                                        "obj",
                                        (object,),
                                        {
                                            "message": type(
                                                "obj",
                                                (object,),
                                                {
                                                    "role": "assistant",
                                                    "content": content,
                                                    "tool_calls": tool_calls,
                                                },
                                            )(),
                                            "finish_reason": finish_reason,
                                        },
                                    )()
                                ]
                                self.usage = usage

                        response = StreamResponse(
                            full_content,
                            finish_reason,
                            usage_data,
                            self._materialize_stream_tool_calls(
                                aggregated_tool_calls
                            ),
                        )
                    else:
                        response = self.client.chat.completions.create(**params)
                        self._update_token_usage(getattr(response, "usage", None))

                self.task_log.log_step(
                    "info",
                    "LLM | Response Status",
                    f"{getattr(response.choices[0], 'finish_reason', 'N/A')}",
                )

                # Check if response was truncated due to length limit
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    # If this is not the last retry, increase max_tokens and retry
                    if attempt < max_retries - 1:
                        # Increase max_tokens by 10%
                        current_max_tokens = int(current_max_tokens * 1.1)
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached",
                            f"Response was truncated due to length limit (attempt {attempt + 1}/{max_retries}). Increasing max_tokens to {current_max_tokens} and retrying...",
                        )
                        await asyncio.sleep(base_wait_time)
                        continue
                    else:
                        # Last retry, return the truncated response instead of raising exception
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached - Returning Truncated Response",
                            f"Response was truncated after {max_retries} attempts. Returning truncated response to allow ReAct loop to continue.",
                        )
                        # Return the truncated response and let the orchestrator handle it
                        return response, messages_history

                # Check if the last 50 characters of the response appear more than 5 times in the response content.
                # If so, treat it as a severe repeat and trigger a retry.
                if hasattr(response.choices[0], "message") and hasattr(
                    response.choices[0].message, "content"
                ):
                    resp_content = response.choices[0].message.content or ""
                else:
                    resp_content = getattr(response.choices[0], "text", "")

                if resp_content and len(resp_content) >= 50:
                    tail_50 = resp_content[-50:]
                    repeat_count = resp_content.count(tail_50)
                    if repeat_count > 5:
                        # If this is not the last retry, retry
                        if attempt < max_retries - 1:
                            self.task_log.log_step(
                                "warning",
                                "LLM | Repeat Detected",
                                f"Severe repeat: the last 50 chars appeared over 5 times (attempt {attempt + 1}/{max_retries}), retrying...",
                            )
                            await asyncio.sleep(base_wait_time)
                            continue
                        else:
                            # Last retry, return anyway
                            self.task_log.log_step(
                                "warning",
                                "LLM | Repeat Detected - Returning Anyway",
                                f"Severe repeat detected after {max_retries} attempts. Returning response anyway.",
                            )

                # Success - return the original messages_history (not the filtered copy)
                # This ensures that the complete conversation history is preserved in logs
                return response, messages_history

            except asyncio.TimeoutError as e:
                if attempt < max_retries - 1:
                    self.task_log.log_step(
                        "warning",
                        "LLM | Timeout Error",
                        f"Timeout error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...",
                    )
                    await asyncio.sleep(base_wait_time)
                    continue
                else:
                    self.task_log.log_step(
                        "error",
                        "LLM | Timeout Error",
                        f"Timeout error after {max_retries} attempts: {str(e)}",
                    )
                    raise e
            except asyncio.CancelledError as e:
                self.task_log.log_step(
                    "error",
                    "LLM | Request Cancelled",
                    f"Request was cancelled: {str(e)}",
                )
                raise e
            except Exception as e:
                connection_error_markers = (
                    "Connection error",
                    "ConnectError",
                    "APIConnectionError",
                    "WinError 10061",
                    "actively refused",
                    "由于目标计算机积极拒绝",
                )
                if any(marker in str(e) for marker in connection_error_markers):
                    self.task_log.log_step(
                        "error",
                        "LLM | Connection Error",
                        "The configured model endpoint is unreachable. Check OPENAI_BASE_URL / proxy service status and network connectivity before retrying.",
                    )
                    raise e

                if "Error code: 400" in str(e) and "longer than the model" in str(e):
                    self.task_log.log_step(
                        "error",
                        "LLM | Context Length Error",
                        f"Error: {str(e)}",
                    )
                    raise e
                else:
                    if attempt < max_retries - 1:
                        self.task_log.log_step(
                            "warning",
                            "LLM | API Error",
                            f"Error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...",
                        )
                        await asyncio.sleep(base_wait_time)
                        continue
                    else:
                        self.task_log.log_step(
                            "error",
                            "LLM | API Error",
                            f"Error after {max_retries} attempts: {str(e)}",
                        )
                        raise e

        # Should never reach here, but just in case
        raise Exception("Unexpected error: retry loop completed without returning")

    def process_llm_response(
        self, llm_response: Any, message_history: List[Dict], agent_type: str = "main"
    ) -> tuple[str, bool, List[Dict]]:
        """Process LLM response"""
        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            self.task_log.log_step(
                "error", "LLM | Response Error", f"Error: {error_msg}"
            )
            return "", True, message_history  # Exit loop, return message_history

        finish_reason = llm_response.choices[0].finish_reason
        message = llm_response.choices[0].message
        assistant_response_text = message.content or ""
        tool_calls = getattr(message, "tool_calls", None) or []

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_response_text,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                self._serialize_tool_call(tool_call) for tool_call in tool_calls
            ]

        # Extract LLM response text
        if finish_reason in {"stop", "tool_calls"} or (
            finish_reason is None and (assistant_response_text or tool_calls)
        ):
            message_history.append(assistant_message)

        elif finish_reason == "length":
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty."
            elif "Context length exceeded" in assistant_response_text:
                # This is the case where context length is exceeded, needs special handling
                self.task_log.log_step(
                    "warning",
                    "LLM | Context Length",
                    "Detected context length exceeded, returning error status",
                )
                assistant_message["content"] = assistant_response_text
                message_history.append(assistant_message)
                return (
                    assistant_response_text,
                    True,
                    message_history,
                )  # Return True to indicate need to exit loop

            # Add assistant response to history
            assistant_message["content"] = assistant_response_text
            message_history.append(assistant_message)

        else:
            raise ValueError(f"Unsupported finish reason: {finish_reason}")

        return assistant_response_text, False, message_history

    def extract_tool_calls_info(
        self, llm_response: Any, assistant_response_text: str
    ) -> List[Dict]:
        """Extract tool call information from LLM response"""
        from ...utils.parsing_utils import parse_llm_response_for_tool_calls

        if (
            llm_response
            and getattr(llm_response, "choices", None)
            and getattr(llm_response.choices[0], "message", None)
            and getattr(llm_response.choices[0].message, "tool_calls", None)
        ):
            return parse_llm_response_for_tool_calls(
                llm_response.choices[0].message.tool_calls
            )

        return parse_llm_response_for_tool_calls(assistant_response_text)

    def update_message_history(
        self, message_history: List[Dict], all_tool_results_content_with_id: List[Tuple]
    ) -> List[Dict]:
        """Update message history with tool calls data (llm client specific)"""
        if message_history and message_history[-1].get("tool_calls"):
            for call_id, tool_result in all_tool_results_content_with_id:
                if tool_result["type"] != "text":
                    continue
                message_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_result["text"],
                    }
                )
            return message_history

        merged_text = "\n".join(
            [
                item[1]["text"]
                for item in all_tool_results_content_with_id
                if item[1]["type"] == "text"
            ]
        )

        message_history.append(
            {
                "role": "user",
                "content": merged_text,
            }
        )

        return message_history

    def generate_agent_system_prompt(self, date: Any, mcp_servers: List[Dict]) -> str:
        return generate_mcp_system_prompt(date, mcp_servers)

    def _estimate_tokens(self, text: str) -> int:
        """Use tiktoken to estimate the number of tokens in text"""
        if not hasattr(self, "encoding"):
            # Initialize tiktoken encoder
            try:
                self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                # If o200k_base is not available, use cl100k_base as fallback
                self.encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # If encoding fails, use simple estimation: approximately 1 token per 4 characters
            self.task_log.log_step(
                "error",
                "LLM | Token Estimation Error",
                f"Error: {str(e)}",
            )
            return len(text) // 4

    def ensure_summary_context(
        self, message_history: list, summary_prompt: str
    ) -> tuple[bool, list]:
        """
        Check if current message_history + summary_prompt will exceed context
        If it will exceed, remove the last assistant-user pair and return False
        Return True to continue, False if messages have been rolled back
        """
        # Get token usage from the last LLM call
        last_prompt_tokens = self.last_call_tokens.get("prompt_tokens", 0)
        last_completion_tokens = self.last_call_tokens.get("completion_tokens", 0)

        buffer_factor = 1.5

        # Calculate token count for summary prompt
        summary_tokens = int(self._estimate_tokens(summary_prompt) * buffer_factor)

        # Calculate token count for the last user message in message_history
        last_user_tokens = 0
        if message_history[-1]["role"] == "user":
            content = message_history[-1]["content"]
            last_user_tokens = int(self._estimate_tokens(content) * buffer_factor)

        # Calculate total token count: last prompt + completion + last user message + summary + reserved response space
        estimated_total = (
            last_prompt_tokens
            + last_completion_tokens
            + last_user_tokens
            + summary_tokens
            + self.max_tokens
            + 1000  # Add 1000 tokens as buffer
        )

        if estimated_total >= self.max_context_length:
            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                "Context limit reached, proceeding to step back and summarize the conversation",
            )

            # Remove the last user message (tool call results)
            if message_history[-1]["role"] == "user":
                message_history.pop()

            # Remove the second-to-last assistant message (tool call request)
            if message_history[-1]["role"] == "assistant":
                message_history.pop()

            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                f"Removed the last assistant-user pair, current message_history length: {len(message_history)}",
            )

            return False, message_history

        self.task_log.log_step(
            "info",
            "LLM | Context Limit Not Reached",
            f"{estimated_total}/{self.max_context_length}",
        )
        return True, message_history

    def format_token_usage_summary(self) -> tuple[List[str], str]:
        """Format token usage statistics, return summary_lines for format_final_summary and log string"""
        token_usage = self.get_token_usage()

        total_input = token_usage.get("total_input_tokens", 0)
        total_output = token_usage.get("total_output_tokens", 0)
        cache_input = token_usage.get("total_cache_input_tokens", 0)

        summary_lines = []
        summary_lines.append("\n" + "-" * 20 + " Token Usage " + "-" * 20)
        summary_lines.append(f"Total Input Tokens: {total_input}")
        summary_lines.append(f"Total Cache Input Tokens: {cache_input}")
        summary_lines.append(f"Total Output Tokens: {total_output}")
        summary_lines.append("-" * (40 + len(" Token Usage ")))
        summary_lines.append("Pricing is disabled - no cost information available")
        summary_lines.append("-" * (40 + len(" Token Usage ")))

        # Generate log string
        log_string = (
            f"[{self.model_name}] Total Input: {total_input}, "
            f"Cache Input: {cache_input}, "
            f"Output: {total_output}"
        )

        return summary_lines, log_string

    def get_token_usage(self):
        return self.token_usage.copy()
