"""Claude Code Router adapter.

This adapter talks to a running `claude-code-router` instance which in turn
proxies requests to other model providers (e.g. OpenRouter's Qwen 3 Coder).
It translates the Anthropic Messages streaming protocol into Claudable's
Message objects and executes the common Claude Code tooling locally.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import httpx

from app.core.terminal_ui import ui
from app.models.messages import Message

from ..base import BaseCLI, CLIType, get_display_path


def _default_router_model() -> str:
    return os.getenv("CLAUDE_CODE_ROUTER_MODEL", "openrouter,qwen/qwen3-coder")


class ClaudeCodeRouterCLI(BaseCLI):
    """HTTP adapter for claude-code-router."""

    _conversation_cache: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, db_session=None):
        super().__init__(CLIType.ROUTER)
        self.db_session = db_session

    # ------------------------------------------------------------------
    # Adapter interface
    async def check_availability(self) -> Dict[str, Any]:
        base_url = self._get_base_url()
        if not base_url:
            return {
                "available": False,
                "configured": False,
                "error": (
                    "Set CLAUDE_CODE_ROUTER_URL to the running claude-code-router "
                    "instance (e.g. http://127.0.0.1:3456)."
                ),
            }

        url = base_url.rstrip("/") + "/health"
        headers = self._build_headers()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return {
                    "available": True,
                    "configured": True,
                    "mode": "HTTP",
                    "base_url": base_url,
                    "models": self.get_supported_models(),
                    "default_models": [_default_router_model()],
                }
            return {
                "available": False,
                "configured": False,
                "error": f"Router healthcheck failed: {response.status_code} {response.text[:200]}",
            }
        except httpx.HTTPError as exc:
            return {
                "available": False,
                "configured": False,
                "error": f"Unable to reach claude-code-router: {exc}",
            }

    async def execute_with_streaming(
        self,
        instruction: str,
        project_path: str,
        session_id: Optional[str] = None,
        log_callback: Optional[Callable[[str], Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        is_initial_prompt: bool = False,
    ) -> AsyncGenerator[Message, None]:
        base_url = self._get_base_url()
        if not base_url:
            yield self._error_message(
                project_path,
                session_id,
                "claude-code-router base URL not configured. Set CLAUDE_CODE_ROUTER_URL.",
            )
            return

        if images:
            ui.warning("Claude Code Router does not yet support image input; ignoring attachments.", "Router")

        from app.services.claude_act import get_system_prompt

        system_prompt = get_system_prompt()
        cli_model = self._get_cli_model_name(model) or _default_router_model()

        project_identifier = Path(project_path).name if project_path else "project"
        active_session_id = (
            session_id or await self.get_session_id(project_identifier) or uuid.uuid4().hex
        )
        history = list(self._conversation_cache.get(active_session_id, []))

        final_instruction = instruction
        if is_initial_prompt:
            final_instruction = self._augment_initial_instruction(instruction, project_path)

        history.append({
            "role": "user",
            "content": [{"type": "text", "text": final_instruction}],
        })

        metadata_user = f"project_{project_identifier}_session_{active_session_id}"

        while True:
            try:
                content_blocks, usage = await self._stream_once(
                    base_url,
                    history,
                    system_prompt,
                    cli_model,
                    metadata_user,
                )
            except Exception as exc:  # pragma: no cover - network failure path
                ui.error(f"Router request failed: {exc}", "Router")
                yield self._error_message(
                    project_path,
                    session_id,
                    f"Router request failed: {exc}",
                )
                return

            text_output = self._collect_text(content_blocks)
            if text_output.strip():
                yield Message(
                    id=str(uuid.uuid4()),
                    project_id=project_path,
                    role="assistant",
                    message_type="chat",
                    content=text_output.strip(),
                    metadata_json={
                        "cli_type": self.cli_type.value,
                        "model": cli_model,
                        "usage": usage,
                    },
                    session_id=session_id,
                    created_at=self._now(),
                )

            tool_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]

            for tool in tool_blocks:
                summary = self._create_tool_summary(tool.get("name"), tool.get("input", {}))
                yield Message(
                    id=str(uuid.uuid4()),
                    project_id=project_path,
                    role="assistant",
                    message_type="tool_use",
                    content=summary,
                    metadata_json={
                        "cli_type": self.cli_type.value,
                        "tool_name": tool.get("name"),
                        "tool_input": tool.get("input"),
                    },
                    session_id=session_id,
                    created_at=self._now(),
                )

            history.append(self._format_assistant_blocks(content_blocks))

            if not tool_blocks:
                break

            for tool in tool_blocks:
                result_text, is_error = await self._execute_tool(tool, project_path)
                history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool.get("id"),
                            "content": result_text,
                            "is_error": is_error,
                        }
                    ],
                })

                if result_text:
                    yield Message(
                        id=str(uuid.uuid4()),
                        project_id=project_path,
                        role="assistant",
                        message_type="tool_result",
                        content=result_text[:4000],
                        metadata_json={
                            "cli_type": self.cli_type.value,
                            "event_type": "tool_result",
                            "tool_name": tool.get("name"),
                            "tool_use_id": tool.get("id"),
                            "is_error": is_error,
                        },
                        session_id=session_id,
                        created_at=self._now(),
                    )

        self._conversation_cache[active_session_id] = history
        await self.set_session_id(project_identifier, active_session_id)

    async def get_session_id(self, project_id: str) -> Optional[str]:
        if not self.db_session:
            return None
        try:
            from app.models.projects import Project

            project = self.db_session.query(Project).filter(Project.id == project_id).first()
            if not project:
                return None
            settings = project.settings or {}
            return settings.get("router_session_id")
        except Exception as exc:  # pragma: no cover - DB failure path
            ui.warning(f"Failed to load router session id: {exc}", "Router")
            return None

    async def set_session_id(self, project_id: str, session_id: str) -> None:
        if not self.db_session:
            return
        try:
            from app.models.projects import Project

            project = self.db_session.query(Project).filter(Project.id == project_id).first()
            if not project:
                return
            settings = project.settings or {}
            settings["router_session_id"] = session_id
            project.settings = settings
            self.db_session.commit()
        except Exception as exc:  # pragma: no cover - DB failure path
            self.db_session.rollback()
            ui.warning(f"Failed to persist router session id: {exc}", "Router")

    # ------------------------------------------------------------------
    # Internal helpers
    def _get_base_url(self) -> Optional[str]:
        """Return the configured router base URL with sensible fallbacks.

        We prefer ``CLAUDE_CODE_ROUTER_URL`` when provided. When it is missing
        (common in new Codespaces), we fall back to the local loopback port or
        the forwarded Codespaces hostname so health checks keep working without
        extra configuration.
        """

        explicit_url = os.getenv("CLAUDE_CODE_ROUTER_URL")
        if explicit_url:
            return explicit_url

        port = os.getenv("CLAUDE_CODE_ROUTER_PORT", "3456")

        codespace = os.getenv("CODESPACE_NAME")
        forwarding_domain = os.getenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
        if codespace and forwarding_domain:
            return f"https://{port}-{codespace}.{forwarding_domain}"

        return f"http://127.0.0.1:{port}"

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"content-type": "application/json"}
        api_key = os.getenv("CLAUDE_CODE_ROUTER_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
            headers.setdefault("authorization", f"Bearer {api_key}")
        return headers

    async def _stream_once(
        self,
        base_url: str,
        history: List[Dict[str, Any]],
        system_prompt: str,
        model: str,
        metadata_user: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        payload = {
            "model": model,
            "messages": history,
            "system": [{"type": "text", "text": system_prompt}],
            "stream": True,
            "metadata": {"user_id": metadata_user},
        }

        url = base_url.rstrip("/") + "/v1/messages"
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    text = await response.aread()
                    raise RuntimeError(
                        f"Router error {response.status_code}: {text.decode(errors='ignore')[:500]}"
                    )

                block_states: Dict[int, Dict[str, Any]] = {}
                usage: Dict[str, Any] = {}

                async for event, data in self._iter_sse(response):
                    if event == "content_block_start":
                        index = data.get("index", 0)
                        block = data.get("content_block", {}) or {}
                        block_type = block.get("type")
                        if block_type == "output_text":
                            block_states[index] = {"type": "text", "text": ""}
                        elif block_type == "tool_use":
                            block_states[index] = {
                                "type": "tool_use",
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input_buffer": "",
                            }
                    elif event == "content_block_delta":
                        index = data.get("index", 0)
                        block = block_states.get(index)
                        if not block:
                            continue
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")
                        if block.get("type") == "text" and delta_type == "output_text_delta":
                            block["text"] = block.get("text", "") + delta.get("text", "")
                        elif block.get("type") == "tool_use" and delta_type == "input_json_delta":
                            block["input_buffer"] = block.get("input_buffer", "") + delta.get("partial_json", "")
                    elif event == "content_block_stop":
                        index = data.get("index", 0)
                        block = block_states.get(index)
                        if not block:
                            continue
                        if block.get("type") == "tool_use":
                            raw = block.pop("input_buffer", "")
                            if raw:
                                try:
                                    block["input"] = json.loads(raw)
                                except json.JSONDecodeError:
                                    block["input"] = {"raw": raw}
                            else:
                                block["input"] = {}
                    elif event == "message_delta":
                        usage.update(data.get("usage", {}))
                    elif event == "error":  # pragma: no cover - propagated error
                        raise RuntimeError(data.get("error", "router stream error"))

                content_blocks: List[Dict[str, Any]] = []
                for index in sorted(block_states.keys()):
                    block = block_states[index]
                    if block.get("type") == "text":
                        content_blocks.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "tool_use":
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input": block.get("input", {}),
                            }
                        )

                return content_blocks, usage

    async def _iter_sse(self, response: httpx.Response):
        event_name: Optional[str] = None
        data_lines: List[str] = []

        async for raw_line in response.aiter_lines():
            if raw_line == "":
                if event_name and data_lines:
                    data = self._parse_sse_data("\n".join(data_lines))
                    yield event_name, data
                event_name = None
                data_lines = []
                continue

            if raw_line.startswith(":"):
                continue
            if raw_line.startswith("event:"):
                event_name = raw_line[len("event:") :].strip()
            elif raw_line.startswith("data:"):
                data_lines.append(raw_line[len("data:") :].strip())

        if event_name and data_lines:
            data = self._parse_sse_data("\n".join(data_lines))
            yield event_name, data

    def _parse_sse_data(self, payload: str) -> Dict[str, Any]:
        try:
            return json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            return {"raw": payload}

    def _collect_text(self, blocks: List[Dict[str, Any]]) -> str:
        return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")

    def _format_assistant_blocks(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for block in blocks:
            if block.get("type") == "text":
                content.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                    }
                )
        return {"role": "assistant", "content": content}

    def _augment_initial_instruction(self, instruction: str, project_path: str) -> str:
        try:
            entries = []
            for root, dirs, files in os.walk(project_path):
                rel_root = os.path.relpath(root, project_path)
                if rel_root == ".":
                    rel_root = ""
                for name in sorted(files):
                    if name.startswith("."):
                        continue
                    full = os.path.join(rel_root, name) if rel_root else name
                    entries.append(full)
                if len(entries) > 200:
                    break
            structure = "\n".join(entries[:200])
            return (
                instruction
                + "\n\n<initial_context>\n"
                + structure
                + "\n</initial_context>"
            )
        except Exception:
            return instruction

    async def _execute_tool(self, tool: Dict[str, Any], project_path: str) -> Tuple[str, bool]:
        name = tool.get("name") or ""
        tool_input = tool.get("input") or {}
        normalized = self._normalize_tool_name(name)

        try:
            if normalized == "Read":
                result = await self._tool_read(project_path, tool_input)
                return result, False
            if normalized in ("Write", "Edit"):
                result = await self._tool_write(project_path, tool_input)
                return result, False
            if normalized == "LS":
                result = await self._tool_ls(project_path, tool_input)
                return result, False
            if normalized == "Glob":
                result = await self._tool_glob(project_path, tool_input)
                return result, False
            if normalized == "Grep":
                result = await self._tool_grep(project_path, tool_input)
                return result, False
            if normalized == "Bash":
                result = await self._tool_bash(project_path, tool_input)
                return result, False
        except Exception as exc:
            return f"Error running tool {name}: {exc}", True

        return f"Tool {name} is not supported yet.", True

    async def _tool_read(self, project_path: str, params: Dict[str, Any]) -> str:
        path = (
            params.get("file_path")
            or params.get("path")
            or params.get("file")
        )
        if not path:
            raise ValueError("read tool missing path")
        abs_path = self._resolve_path(project_path, path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(path)
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content[:200000]

    async def _tool_write(self, project_path: str, params: Dict[str, Any]) -> str:
        path = (
            params.get("file_path")
            or params.get("path")
            or params.get("file")
        )
        if not path:
            raise ValueError("write tool missing path")
        abs_path = self._resolve_path(project_path, path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        content = (
            params.get("content")
            or params.get("text")
            or params.get("code")
            or params.get("new_content")
        )
        if content is None:
            if params.get("old_content") and params.get("replacement"):
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    existing = f.read()
                content = existing.replace(params["old_content"], params["replacement"])
            else:
                raise ValueError("write tool missing content")

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Wrote {len(content)} characters to {get_display_path(abs_path)}"

    async def _tool_ls(self, project_path: str, params: Dict[str, Any]) -> str:
        path = params.get("path") or params.get("directory") or "."
        abs_path = self._resolve_path(project_path, path)
        if not os.path.isdir(abs_path):
            raise NotADirectoryError(path)
        entries = sorted(os.listdir(abs_path))
        display = [entry for entry in entries if not entry.startswith(".git")]
        return "\n".join(display[:200])

    async def _tool_glob(self, project_path: str, params: Dict[str, Any]) -> str:
        import glob

        pattern = params.get("pattern") or params.get("globPattern")
        if not pattern:
            raise ValueError("glob tool missing pattern")
        full_pattern = os.path.join(project_path, pattern)
        matches = [
            get_display_path(match)
            for match in glob.glob(full_pattern, recursive=True)
        ]
        return "\n".join(matches[:200]) if matches else "No matches found"

    async def _tool_grep(self, project_path: str, params: Dict[str, Any]) -> str:
        pattern = params.get("pattern") or params.get("query")
        if not pattern:
            raise ValueError("grep tool missing pattern")
        path = params.get("path") or params.get("file")
        results: List[str] = []
        regex = re.compile(pattern)

        if path:
            abs_path = self._resolve_path(project_path, path)
            if os.path.isdir(abs_path):
                for root, _, files in os.walk(abs_path):
                    for name in files:
                        target = os.path.join(root, name)
                        results.extend(self._grep_file(target, regex))
            else:
                results.extend(self._grep_file(abs_path, regex))
        else:
            for root, _, files in os.walk(project_path):
                for name in files:
                    target = os.path.join(root, name)
                    results.extend(self._grep_file(target, regex))

        if not results:
            return "No matches found"
        return "\n".join(results[:200])

    def _grep_file(self, path: str, regex: re.Pattern[str]) -> List[str]:
        matches: List[str] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    if regex.search(line):
                        matches.append(f"{get_display_path(path)}:{idx}: {line.strip()}")
        except Exception:
            pass
        return matches

    async def _tool_bash(self, project_path: str, params: Dict[str, Any]) -> str:
        command = params.get("command") or params.get("cmd")
        if not command:
            raise ValueError("bash tool missing command")

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=project_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = ""
        if stdout:
            output += stdout.decode(errors="ignore")
        if stderr:
            output += "\n" + stderr.decode(errors="ignore")
        return output.strip()[:8000] or f"Command exited with code {process.returncode}"

    def _resolve_path(self, project_path: str, raw_path: str) -> str:
        base = os.path.abspath(project_path or ".")
        joined = os.path.abspath(os.path.join(base, raw_path))
        if not joined.startswith(base):
            raise ValueError("Path escapes project directory")
        return joined

    def _now(self):
        from datetime import datetime

        return datetime.utcnow()

    def _error_message(self, project_path: str, session_id: Optional[str], content: str) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            project_id=project_path,
            role="assistant",
            message_type="error",
            content=content,
            metadata_json={"cli_type": self.cli_type.value},
            session_id=session_id,
            created_at=self._now(),
        )


__all__ = ["ClaudeCodeRouterCLI"]
