from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from gateway_guardrails import GatewayInfrastructure


logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]

RELATIVE_REFERENCE_PATTERN = re.compile(
    r"\b("
    r"it|this|that|these|those|"
    r"this\s+(?:chart|figure|table|diagram|image|csv|file|data|visual)|"
    r"that\s+(?:chart|figure|table|diagram|image|csv|file|data|visual)|"
    r"the\s+(?:chart|figure|table|diagram|image|csv|file|data|visual)\s+(?:above|before|shown|mentioned)|"
    r"same\s+(?:chart|figure|table|diagram|image|csv|file|data|visual)"
    r")\b",
    flags=re.IGNORECASE,
)

ASSET_METADATA_KEYS = (
    "image_path",
    "image_local_path",
    "csv_path",
    "table_path",
    "asset_path",
    "file_path",
)

REDACTION_TOKEN_PATTERN = re.compile(r"\[REDACTED_[A-Z_]+\]")


@dataclass
class ConversationTurn:
    role: Role
    content: str
    asset_paths: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)


class MultimodalConversationManager:
    """Asset-aware conversation memory with a bounded prompt window."""

    def __init__(
        self,
        session_id: str = "default",
        max_turns: int = 6,
        system_instructions: str | None = None,
        storage_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.max_turns = max(1, int(max_turns))
        self.storage_path = Path(
            storage_path
            or os.getenv("RAG_CHAT_HISTORY_PATH", "data_cache/chat_history.json")
        )
        self.system_instructions = system_instructions or (
            "You are a grounded multimodal RAG assistant. Use the compressed retrieved context as factual evidence. "
            "Use conversation history only to resolve references and preserve continuity. Inspect any active visual "
            "or data file paths when they are supplied."
        )
        self._sessions: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._lock = threading.RLock()
        self._gateway = GatewayInfrastructure(request_cap=1_000_000)
        self._load_from_disk()

    def get_history(self, session_id: str | None = None) -> list[ConversationTurn]:
        resolved_session = session_id or self.session_id
        with self._lock:
            history = list(self._sessions.get(resolved_session, ()))
        logger.info(
            "Conversation memory read session_id=%s held_turns=%s",
            resolved_session,
            len(history),
        )
        return history

    def get_optimized_history(self, session_id: str | None = None, max_turns: int = 3) -> list[dict[str, Any]]:
        history = self.get_history(session_id)
        window = history[-max(max_turns, 0) * 2 :] if max_turns else []
        return [asdict(turn) for turn in window]

    def get_full_history(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return [asdict(turn) for turn in self.get_history(session_id)]

    def clear(self, session_id: str | None = None) -> None:
        resolved_session = session_id or self.session_id
        with self._lock:
            self._sessions.pop(resolved_session, None)
            self._save_to_disk()
        logger.info("Conversation memory cleared session_id=%s", resolved_session)

    def clear_history(self, session_id: str | None = None) -> None:
        self.clear(session_id)

    def record_user_turn(
        self,
        user_query: str,
        *,
        asset_paths: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._append_turn(
            ConversationTurn(
                role="user",
                content=self._mask_pii(user_query),
                asset_paths=self._dedupe_paths(asset_paths or []),
            ),
            session_id=session_id,
        )

    def update_after_generation(
        self,
        user_query: str,
        assistant_response: str,
        compressed_context_chunks: list[dict[str, Any]] | None = None,
        *,
        active_asset_paths: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Store a completed user/assistant exchange and pin active assets to the assistant turn."""

        chunk_assets = self.extract_asset_paths_from_chunks(compressed_context_chunks or [])
        pinned_assets = self._dedupe_paths([*(active_asset_paths or []), *chunk_assets])
        self.record_user_turn(user_query, session_id=session_id)
        self._append_turn(
            ConversationTurn(
                role="assistant",
                content=self._mask_pii(assistant_response),
                asset_paths=pinned_assets,
                sources=list(compressed_context_chunks or []),
            ),
            session_id=session_id,
        )
        resolved_session = session_id or self.session_id
        logger.info(
            "Conversation memory updated session_id=%s held_turns=%s pinned_assets=%s",
            resolved_session,
            len(self.get_history(resolved_session)),
            len(pinned_assets),
        )

    def update_history(self, session_id: str, user_query: str, ai_response: str) -> None:
        self.update_after_generation(user_query, ai_response, [], session_id=session_id)

    def update_session_state(
        self,
        *,
        query: str = "",
        response: str,
        chunks: list[dict[str, Any]] | None = None,
        active_asset_paths: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.update_after_generation(
            query,
            response,
            chunks or [],
            active_asset_paths=active_asset_paths,
            session_id=session_id,
        )

    def attach_sources(self, session_id: str, sources: list[dict[str, Any]]) -> None:
        with self._lock:
            history = self._sessions.get(session_id)
            if not history:
                return
            last_turn = history[-1]
            if last_turn.role != "assistant":
                return
            last_turn.sources = list(sources or [])
            last_turn.asset_paths = self._dedupe_paths(
                [*last_turn.asset_paths, *self.extract_asset_paths_from_chunks(sources or [])]
            )
            held_turns = len(history)
            self._save_to_disk()
        logger.info(
            "Conversation memory sources attached session_id=%s held_turns=%s source_chunks=%s assets=%s",
            session_id,
            held_turns,
            len(sources or []),
            len(last_turn.asset_paths),
        )

    def compile_generator_input(
        self,
        current_query: str,
        compressed_context_chunks: list[dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Return a provider-neutral LLM payload with system instructions, transcript,
        compressed context, and active multimodal asset paths.
        """

        resolved_session = session_id or self.session_id
        history = self.get_history(resolved_session)
        current_assets = self.extract_asset_paths_from_chunks(compressed_context_chunks)
        historical_assets = self._historical_assets_for_query(current_query, history)
        active_asset_paths = self._dedupe_paths([*current_assets, *historical_assets])
        transcript = self._mask_pii(self._format_transcript(history))
        context_text = self._format_context(compressed_context_chunks)

        logger.info(
            "Generator input compiled session_id=%s held_turns=%s context_chunks=%s active_assets=%s",
            resolved_session,
            len(history),
            len(compressed_context_chunks or []),
            len(active_asset_paths),
        )

        return {
            "system": self.system_instructions,
            "messages": [
                {"role": "system", "content": self.system_instructions},
                {
                    "role": "user",
                    "content": (
                        f"[CHAT HISTORY]\n{transcript or '(none)'}\n\n"
                        f"[COMPRESSED RETRIEVED CONTEXT]\n{context_text or '(none)'}\n\n"
                        f"[ACTIVE VISUAL/DATA FILE PATHS]\n"
                        f"{self._format_asset_paths(active_asset_paths) or '(none)'}\n\n"
                        f"[CURRENT USER QUERY]\n{self._mask_pii(current_query)}"
                    ),
                },
            ],
            "chat_history_transcript": transcript,
            "compressed_context_text": context_text,
            "active_asset_paths": active_asset_paths,
            "context_chunks": compressed_context_chunks,
        }

    def redact_condensed_payload(self, condensed_query: Any, *source_payloads: Any) -> str:
        """
        Apply Layer 2 PII redaction to a condensed query and preserve evidence that
        upstream source payloads contained PII even when an LLM paraphrases it away.
        """

        redacted_query = self._mask_pii(condensed_query)
        source_markers = self.redaction_markers_for_payloads(*source_payloads)
        if not source_markers:
            return redacted_query

        present_markers = set(REDACTION_TOKEN_PATTERN.findall(redacted_query))
        missing_markers = [marker for marker in source_markers if marker not in present_markers]
        if not missing_markers:
            return redacted_query

        marker_block = " ".join(missing_markers)
        return f"{redacted_query}\nLayer 2 redaction markers from source context: {marker_block}".strip()

    def redaction_markers_for_payloads(self, *payloads: Any) -> list[str]:
        markers: list[str] = []
        for payload in payloads:
            redacted_payload = self._mask_pii(self._stringify_payload(payload))
            for marker in REDACTION_TOKEN_PATTERN.findall(redacted_payload):
                if marker not in markers:
                    markers.append(marker)
        return markers

    def extract_asset_paths_from_chunks(self, chunks: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            for key in ASSET_METADATA_KEYS:
                for value in (chunk.get(key), metadata.get(key)):
                    if isinstance(value, str) and value.strip():
                        paths.append(value.strip())
            text = self._chunk_text(chunk)
            paths.extend(self._extract_paths_from_text(text))
        return self._dedupe_paths(paths)

    def to_serializable_history(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return [asdict(turn) for turn in self.get_history(session_id)]

    def _append_turn(self, turn: ConversationTurn, *, session_id: str | None = None) -> None:
        resolved_session = session_id or self.session_id
        with self._lock:
            self._sessions[resolved_session].append(turn)
            held_turns = len(self._sessions[resolved_session])
            self._save_to_disk()
        logger.info(
            "Conversation memory append session_id=%s role=%s held_turns=%s assets=%s",
            resolved_session,
            turn.role,
            held_turns,
            len(turn.asset_paths),
        )

    def _historical_assets_for_query(self, query: str, history: list[ConversationTurn]) -> list[str]:
        if not RELATIVE_REFERENCE_PATTERN.search(query or ""):
            return []
        assets: list[str] = []
        for turn in reversed(history[-4:]):
            if turn.asset_paths:
                assets.extend(turn.asset_paths)
                break
        return self._dedupe_paths(assets)

    @staticmethod
    def _format_transcript(history: list[ConversationTurn]) -> str:
        return "\n".join(f"{turn.role.title()}: {turn.content}" for turn in history if turn.content)

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, chunk in enumerate(chunks or [], start=1):
            text = MultimodalConversationManager._chunk_text(chunk)
            metadata = chunk.get("metadata") if isinstance(chunk, dict) and isinstance(chunk.get("metadata"), dict) else {}
            source = chunk.get("source") or metadata.get("source") or metadata.get("source_file") or "unknown"
            page = metadata.get("page_number") or metadata.get("page") or metadata.get("source_page")
            header = f"Chunk {index} | Source: {source}"
            if page:
                header = f"{header} | Page: {page}"
            blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks).strip()

    @staticmethod
    def _format_asset_paths(paths: list[str]) -> str:
        return "\n".join(f"- {path}" for path in paths)

    @staticmethod
    def _chunk_text(chunk: dict[str, Any]) -> str:
        if not isinstance(chunk, dict):
            return ""
        for key in ("content", "text", "page_content"):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_paths_from_text(text: str) -> list[str]:
        if not text:
            return []
        markdown_paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        raw_paths = re.findall(r"\b\S+\.(?:png|jpg|jpeg|webp|gif|csv|xlsx|xls)\b", text, flags=re.IGNORECASE)
        return [*markdown_paths, *raw_paths]

    @staticmethod
    def _stringify_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, ConversationTurn):
            return payload.content
        if isinstance(payload, dict):
            return " ".join(str(value or "") for value in payload.values())
        if isinstance(payload, (list, tuple, set)):
            return "\n".join(MultimodalConversationManager._stringify_payload(item) for item in payload)
        return str(payload)

    @staticmethod
    def _dedupe_paths(paths: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for path in paths:
            normalized = str(path or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(normalized)
        return deduped

    def _load_from_disk(self) -> None:
        try:
            if not self.storage_path.is_file():
                return
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            for session_id, turns in raw.items():
                if not isinstance(turns, list):
                    continue
                loaded_turns: list[ConversationTurn] = []
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    role = turn.get("role")
                    if role not in {"user", "assistant"}:
                        continue
                    loaded_turns.append(
                        ConversationTurn(
                            role=role,
                            content=self._mask_pii(turn.get("content") or ""),
                            asset_paths=list(turn.get("asset_paths") or []),
                            sources=list(turn.get("sources") or []),
                        )
                    )
                self._sessions[str(session_id)] = loaded_turns
        except Exception as exc:
            logger.warning("Could not load chat history from %s: %s", self.storage_path, exc)

    def _save_to_disk(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                session_id: [asdict(turn) for turn in turns]
                for session_id, turns in self._sessions.items()
            }
            self.storage_path.write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save chat history to %s: %s", self.storage_path, exc)

    def _mask_pii(self, text: Any) -> str:
        return self._gateway.mask_pii(str(text or "").strip())
