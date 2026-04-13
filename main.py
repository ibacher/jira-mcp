import json
import os
import re
import sys
from base64 import b64encode
from io import TextIOWrapper
from typing import Any

import aiohttp
import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.stdio import stdio_server

mcp = FastMCP("jira-mcp")

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://openmrs.atlassian.net")


def _auth_header() -> str:
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")
    if not email or not token:
        raise ValueError(
            "JIRA_EMAIL and JIRA_API_TOKEN environment variables must be set"
        )
    credentials = b64encode(f"{email}:{token}".encode()).decode()
    return f"Basic {credentials}"


_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    """Return a shared ClientSession, creating one if needed."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def _request(method: str, path: str, **kwargs) -> tuple[int, Any]:
    """Make an authenticated request to the Jira REST API.

    Returns (status_code, parsed_json_or_text).
    """
    url = f"{JIRA_BASE_URL}{path}"
    headers = kwargs.pop("headers", {})
    headers["Accept"] = "application/json"
    headers["Authorization"] = _auth_header()

    timeout = aiohttp.ClientTimeout(total=kwargs.pop("timeout", 30))

    session = _get_session()
    async with session.request(
        method, url, headers=headers, timeout=timeout, **kwargs
    ) as resp:
        status = resp.status
        content_type = resp.content_type or ""
        if "json" in content_type:
            body = await resp.json()
        else:
            body = await resp.text()
        return status, body


def _error_message(status: int, body: Any) -> str:
    """Extract a readable error from a failed Jira response."""
    if isinstance(body, str):
        return f"Jira error (HTTP {status}): {body}"
    parts = []
    if "errorMessages" in body:
        parts.extend(body["errorMessages"])
    if "errors" in body and isinstance(body["errors"], dict):
        for field, msg in body["errors"].items():
            parts.append(f"{field}: {msg}")
    if parts:
        return f"Jira error (HTTP {status}): " + "; ".join(parts)
    return f"Jira error (HTTP {status}): {json.dumps(body)}"


def _is_ok(status: int) -> bool:
    return 200 <= status < 300


# ---------------------------------------------------------------------------
# Markdown -> ADF conversion
# ---------------------------------------------------------------------------


def _markdown_to_adf(markdown: str) -> dict:
    """Convert a Markdown string to a minimal Atlassian Document Format document.

    Handles paragraphs, headings (# through ######), bullet lists (- or *),
    ordered lists (1. etc.), bold (**), italic (*/_), inline code (`),
    code blocks (```), and links ([text](url)).
    """
    # Clients may send literal "\n" escape sequences (backslash + n) instead
    # of real newlines — the tool docstrings even recommend this to avoid
    # breaking JSON-RPC framing.  Normalise them before parsing.
    markdown = markdown.replace("\\n", "\n")
    lines = markdown.split("\n")
    doc_content: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            language = line[3:].strip() or None
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            node: dict = {
                "type": "codeBlock",
                "content": [{"type": "text", "text": "\n".join(code_lines)}],
            }
            if language:
                node["attrs"] = {"language": language}
            doc_content.append(node)
            continue

        # Blank line — skip
        if not line.strip():
            i += 1
            continue

        # Heading
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            level = len(heading_match.group(1))
            doc_content.append(
                {
                    "type": "heading",
                    "attrs": {"level": level},
                    "content": _inline_markup(heading_match.group(2)),
                }
            )
            i += 1
            continue

        # Unordered list
        if re.match(r"^[\-\*]\s+", line):
            items: list[dict] = []
            while i < len(lines) and re.match(r"^[\-\*]\s+", lines[i]):
                text = re.sub(r"^[\-\*]\s+", "", lines[i])
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": _inline_markup(text)}
                        ],
                    }
                )
                i += 1
            doc_content.append({"type": "bulletList", "content": items})
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                text = re.sub(r"^\d+\.\s+", "", lines[i])
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": _inline_markup(text)}
                        ],
                    }
                )
                i += 1
            doc_content.append({"type": "orderedList", "content": items})
            continue

        # Plain paragraph
        doc_content.append(
            {
                "type": "paragraph",
                "content": _inline_markup(line),
            }
        )
        i += 1

    return {"version": 1, "type": "doc", "content": doc_content}


_INLINE_PATTERN = re.compile(
    r"(`[^`]+`)"  # inline code
    r"|(\*\*[^*]+\*\*)"  # bold
    r"|(\*[^*]+\*)"  # italic with *
    r"|(?<!\w)(_[^_]+_)(?!\w)"  # italic with _ (not mid-word)
    r"|(\[[^\]]+\]\([^)]+\))"  # link
)


def _inline_markup(text: str) -> list[dict]:
    """Parse inline Markdown into ADF text/mark nodes."""
    nodes: list[dict] = []
    pos = 0

    for m in _INLINE_PATTERN.finditer(text):
        if m.start() > pos:
            nodes.append({"type": "text", "text": text[pos : m.start()]})

        matched = m.group()
        if matched.startswith("`"):
            nodes.append(
                {
                    "type": "text",
                    "text": matched[1:-1],
                    "marks": [{"type": "code"}],
                }
            )
        elif matched.startswith("**"):
            nodes.append(
                {
                    "type": "text",
                    "text": matched[2:-2],
                    "marks": [{"type": "strong"}],
                }
            )
        elif matched.startswith("["):
            link_match = re.match(r"\[([^\]]+)\]\(([^)]+)\)", matched)
            if link_match:
                nodes.append(
                    {
                        "type": "text",
                        "text": link_match.group(1),
                        "marks": [
                            {"type": "link", "attrs": {"href": link_match.group(2)}}
                        ],
                    }
                )
        elif matched.startswith("*") or matched.startswith("_"):
            nodes.append(
                {
                    "type": "text",
                    "text": matched[1:-1],
                    "marks": [{"type": "em"}],
                }
            )

        pos = m.end()

    if pos < len(text):
        nodes.append({"type": "text", "text": text[pos:]})

    if not nodes:
        nodes.append({"type": "text", "text": text})

    return nodes


def _adf_to_plain_text(node: dict | list | str | None) -> str:
    """Recursively extract plain text from an ADF document."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_adf_to_plain_text(n) for n in node)
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        parts = []
        for child in node.get("content", []):
            parts.append(_adf_to_plain_text(child))
        if node.get("type") in ("paragraph", "heading", "codeBlock", "listItem"):
            return "".join(parts) + "\n"
        return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def createJiraIssue(
    projectKey: str,
    summary: str,
    issueType: str,
    description: str | None = None,
    priority: str | None = None,
    labels: list[str] | None = None,
    assigneeAccountId: str | None = None,
) -> str:
    """Create a Jira issue in the OpenMRS Jira instance.

    Args:
        projectKey: The Jira project key, e.g. "OCLOMRS"
        summary: One-line summary / title of the issue
        issueType: Issue type name, e.g. "Bug", "Task", "Story"
        description: Optional description in Markdown (converted to ADF).
            IMPORTANT: use literal \\n escape sequences for newlines, not
            actual line breaks, to avoid breaking the JSON-RPC transport.
        priority: Optional priority name, e.g. "High", "Medium", "Low"
        labels: Optional list of label strings
        assigneeAccountId: Optional Jira account ID for the assignee
    """
    fields: dict = {
        "project": {"key": projectKey},
        "summary": summary,
        "issuetype": {"name": issueType},
    }
    if description:
        fields["description"] = _markdown_to_adf(description)
    if priority:
        fields["priority"] = {"name": priority}
    if labels:
        fields["labels"] = labels
    if assigneeAccountId:
        fields["assignee"] = {"accountId": assigneeAccountId}

    status, body = await _request("POST", "/rest/api/3/issue", json={"fields": fields})
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    key = body["key"]
    return f"Created {key}: {JIRA_BASE_URL}/browse/{key}"


@mcp.tool()
async def getJiraIssue(issueIdOrKey: str, fields: list[str] | None = None) -> str:
    """Fetch a Jira issue by key or ID.

    Args:
        issueIdOrKey: Issue key (e.g. "OCLOMRS-123") or numeric ID
        fields: Optional list of field names to return, e.g. ["summary", "status"]
    """
    params = {}
    if fields:
        params["fields"] = ",".join(fields)

    status, body = await _request(
        "GET", f"/rest/api/3/issue/{issueIdOrKey}", params=params
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    f = body["fields"]
    lines = [
        f"**{body['key']}**: {f.get('summary', '(no summary)')}",
        f"URL: {JIRA_BASE_URL}/browse/{body['key']}",
        f"Status: {f['status']['name']}" if "status" in f else None,
        f"Type: {f['issuetype']['name']}" if "issuetype" in f else None,
        f"Priority: {f['priority']['name']}" if f.get("priority") else None,
        f"Assignee: {f['assignee']['displayName']}" if f.get("assignee") else None,
        f"Labels: {', '.join(f['labels'])}" if f.get("labels") else None,
    ]
    desc = f.get("description")
    if desc:
        lines.append(f"Description:\n{_adf_to_plain_text(desc)}")

    return "\n".join(line for line in lines if line is not None)


@mcp.tool()
async def editJiraIssue(
    issueIdOrKey: str,
    summary: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    labels: list[str] | None = None,
    assigneeAccountId: str | None = None,
    issueType: str | None = None,
) -> str:
    """Edit fields on an existing Jira issue.

    Args:
        issueIdOrKey: Issue key (e.g. "OCLOMRS-123") or numeric ID
        summary: New summary / title
        description: New description in Markdown (converted to ADF).
            IMPORTANT: use literal \\n escape sequences for newlines, not
            actual line breaks, to avoid breaking the JSON-RPC transport.
        priority: New priority name, e.g. "High", "Medium", "Low"
        labels: New set of labels (replaces existing labels)
        assigneeAccountId: Jira account ID for the assignee
        issueType: New issue type name, e.g. "Bug", "Task", "Story"
    """
    fields: dict = {}
    if summary is not None:
        fields["summary"] = summary
    if description is not None:
        fields["description"] = _markdown_to_adf(description)
    if priority is not None:
        fields["priority"] = {"name": priority}
    if labels is not None:
        fields["labels"] = labels
    if assigneeAccountId is not None:
        fields["assignee"] = {"accountId": assigneeAccountId}
    if issueType is not None:
        fields["issuetype"] = {"name": issueType}

    if not fields:
        return "No fields provided to update."

    status, body = await _request(
        "PUT", f"/rest/api/3/issue/{issueIdOrKey}", json={"fields": fields}
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    return f"Updated {issueIdOrKey}: {JIRA_BASE_URL}/browse/{issueIdOrKey}"


_JIRA_PAGE_SIZE = 50


@mcp.tool()
async def searchJiraIssues(
    jql: str,
    maxResults: int = 25,
    fields: list[str] | None = None,
) -> str:
    """Search for Jira issues using JQL.

    Args:
        jql: JQL query string, e.g. "project = OCLOMRS AND status = Open"
        maxResults: Maximum number of results to return (default 25).
            Automatically paginates if this exceeds Jira's per-page limit.
        fields: Optional list of fields to return,
            e.g. ["summary", "status", "assignee"]
    """
    issues: list[dict] = []
    total = 0
    start_at = 0

    while len(issues) < maxResults:
        page_size = min(_JIRA_PAGE_SIZE, maxResults - len(issues))
        params: dict = {
            "jql": jql,
            "maxResults": page_size,
            "startAt": start_at,
        }
        if fields:
            params["fields"] = ",".join(fields)

        status, body = await _request("GET", "/rest/api/3/search/jql", params=params)
        if not _is_ok(status):
            raise ToolError(_error_message(status, body))

        total = body.get("total", 0)
        page = body.get("issues", [])
        issues.extend(page)

        # Stop if Jira returned fewer than requested (last page).
        if len(page) < page_size:
            break
        start_at += len(page)

    lines = [f"Found {total} issue(s) (showing {len(issues)}):"]
    for issue in issues:
        f = issue["fields"]
        issue_status = f["status"]["name"] if "status" in f else "?"
        assignee = f["assignee"]["displayName"] if f.get("assignee") else "Unassigned"
        summary = f.get("summary", "")
        lines.append(
            f"- **{issue['key']}** [{issue_status}] {summary} (Assignee: {assignee})"
        )

    return "\n".join(lines)


@mcp.tool()
async def addCommentToJiraIssue(issueIdOrKey: str, body: str) -> str:
    """Add a comment to a Jira issue.

    Args:
        issueIdOrKey: Issue key (e.g. "OCLOMRS-123") or numeric ID
        body: Comment text in Markdown (converted to ADF).
            IMPORTANT: use literal \\n escape sequences for newlines, not
            actual line breaks, to avoid breaking the JSON-RPC transport.
    """
    status, resp_body = await _request(
        "POST",
        f"/rest/api/3/issue/{issueIdOrKey}/comment",
        json={"body": _markdown_to_adf(body)},
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, resp_body))

    return f"Comment added (id: {resp_body['id']}) to {issueIdOrKey}"


@mcp.tool()
async def getTransitionsForJiraIssue(issueIdOrKey: str) -> str:
    """List the available workflow transitions for a Jira issue.

    Args:
        issueIdOrKey: Issue key (e.g. "OCLOMRS-123") or numeric ID
    """
    status, body = await _request(
        "GET", f"/rest/api/3/issue/{issueIdOrKey}/transitions"
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    transitions = body.get("transitions", [])
    if not transitions:
        return f"No transitions available for {issueIdOrKey}."

    lines = [f"Available transitions for {issueIdOrKey}:"]
    for t in transitions:
        lines.append(f'- id={t["id"]} name="{t["name"]}" -> {t["to"]["name"]}')
    return "\n".join(lines)


@mcp.tool()
async def transitionJiraIssue(issueIdOrKey: str, transitionId: str) -> str:
    """Transition a Jira issue to a new status.

    Use getTransitionsForJiraIssue first to find the available transition IDs.

    Args:
        issueIdOrKey: Issue key (e.g. "OCLOMRS-123") or numeric ID
        transitionId: The ID of the transition to execute
            (from getTransitionsForJiraIssue)
    """
    status, body = await _request(
        "POST",
        f"/rest/api/3/issue/{issueIdOrKey}/transitions",
        json={"transition": {"id": transitionId}},
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    return f"Transitioned {issueIdOrKey} (transition id {transitionId})"


@mcp.tool()
async def getMyself() -> str:
    """Get the currently authenticated Jira user's account ID and display name."""
    status, body = await _request("GET", "/rest/api/3/myself")
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    return (
        f"accountId: {body['accountId']}\n"
        f"displayName: {body.get('displayName', '(unknown)')}\n"
        f"emailAddress: {body.get('emailAddress', '(hidden)')}"
    )


@mcp.tool()
async def getVisibleJiraProjects(maxResults: int = 50) -> str:
    """List Jira projects visible to the authenticated user.

    Args:
        maxResults: Maximum number of projects to return (default 50)
    """
    status, body = await _request(
        "GET", "/rest/api/3/project/search", params={"maxResults": maxResults}
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    projects = body.get("values", [])
    lines = [f"Found {len(projects)} project(s):"]
    for p in projects:
        lines.append(f"- **{p['key']}**: {p['name']}")
    return "\n".join(lines)


@mcp.tool()
async def getJiraIssueTypeMetaWithFields(projectIdOrKey: str, issueTypeId: str) -> str:
    """Get the create metadata for an issue type in a project.

    Args:
        projectIdOrKey: Project key (e.g. "OCLOMRS") or numeric project ID
        issueTypeId: Numeric issue type ID (get these from the project metadata)
    """
    status, body = await _request(
        "GET",
        f"/rest/api/3/issue/createmeta/{projectIdOrKey}/issuetypes/{issueTypeId}",
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    fields_list = body.get("values", body.get("fields", []))

    # Dict of field objects (older API shape)
    if isinstance(fields_list, dict):
        lines = [f"Fields for project {projectIdOrKey}, issue type {issueTypeId}:"]
        for key, meta in fields_list.items():
            required = " (required)" if meta.get("required") else ""
            lines.append(f"- {key}: {meta.get('name', key)}{required}")
        return "\n".join(lines)

    # Paginated list shape
    lines = [f"Fields for project {projectIdOrKey}, issue type {issueTypeId}:"]
    for field in fields_list:
        required = " (required)" if field.get("required") else ""
        field_id = field.get("fieldId", field.get("key", "?"))
        name = field.get("name", field_id)
        lines.append(f"- {field_id}: {name}{required}")
    return "\n".join(lines)


class _JsonLineBufferedStdin:
    """Wraps an async text stream and buffers lines until they form valid JSON.

    The MCP stdio transport expects one JSON-RPC message per line, but some
    clients (notably Claude Desktop) may send multiline strings with literal
    newlines, splitting a single JSON message across multiple lines.  This
    wrapper accumulates lines until the buffer parses as valid JSON, then
    re-serializes the message onto a single line (with newlines properly
    escaped) before yielding it.
    """

    def __init__(self, raw_stdin: Any):
        self._stdin = raw_stdin
        self._buffer = ""

    def __aiter__(self) -> "_JsonLineBufferedStdin":
        return self

    @staticmethod
    def _try_parse(text: str) -> dict | None:
        """Try to parse text as JSON, tolerating literal control characters."""
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            return None

    async def __anext__(self) -> str:
        async for line in self._stdin:
            if not self._buffer:
                # Fast path: try the line on its own first with the strict
                # parser — if it passes, it's already well-formed.
                try:
                    json.loads(line)
                    return line
                except json.JSONDecodeError:
                    self._buffer = line
            else:
                self._buffer += line

            parsed = self._try_parse(self._buffer)
            if parsed is not None:
                self._buffer = ""
                # Re-serialize so literal newlines become \\n escapes
                # and the transport sees a single valid JSON line.
                return json.dumps(parsed, ensure_ascii=False) + "\n"

        # stdin exhausted
        if self._buffer:
            remaining = self._buffer
            self._buffer = ""
            return remaining
        raise StopAsyncIteration


async def _run_stdio():
    raw_stdin = anyio.wrap_file(
        TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    )
    buffered = _JsonLineBufferedStdin(raw_stdin)
    streams = stdio_server(
        stdin=buffered,  # ty: ignore[invalid-argument-type]
    )
    async with streams as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


if __name__ == "__main__":
    anyio.run(_run_stdio)
