import json
import os
import re
import sys
from base64 import b64encode
from io import TextIOWrapper
from typing import Annotated, Any

import aiohttp
import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.stdio import stdio_server
from pydantic import Field

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


async def _close_session() -> None:
    """Close the shared ClientSession if one is open."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


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
    try:
        async with session.request(
            method, url, headers=headers, timeout=timeout, **kwargs
        ) as resp:
            status = resp.status
            content_type = resp.content_type or ""
            if "json" in content_type:
                try:
                    body = await resp.json()
                except (
                    aiohttp.ClientPayloadError,
                    aiohttp.ContentTypeError,
                    json.JSONDecodeError,
                ) as e:
                    # Jira claimed JSON but sent something else (e.g. an HTML
                    # error page from a proxy). Surface it rather than letting
                    # a downstream KeyError mask the real cause.
                    text = await resp.text()
                    raise ToolError(
                        f"Jira returned an unparseable response (HTTP {status}): "
                        f"{text[:500]}"
                    ) from e
            else:
                body = await resp.text()
            return status, body
    # In Python 3.11+ asyncio.TimeoutError is an alias of the builtin, which the
    # ClientTimeout above raises when the request exceeds its deadline.
    except TimeoutError as e:
        raise ToolError(
            f"Request to Jira timed out after {timeout.total}s: {method} {path}"
        ) from e
    except aiohttp.ClientError as e:
        raise ToolError(f"Could not reach Jira ({JIRA_BASE_URL}): {e}") from e


def _error_message(status: int, body: Any) -> str:
    """Extract a readable error from a failed Jira response."""
    if isinstance(body, str):
        return f"Jira error (HTTP {status}): {body}"
    if not isinstance(body, dict):
        # Some Jira endpoints return a JSON array (or null) on error; the
        # field-extraction below assumes a dict, so fall back to raw JSON.
        return f"Jira error (HTTP {status}): {json.dumps(body)}"
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
    r"|(\*(?!\s)[^*]+(?<!\s)\*)"  # italic with * (markers must abut non-space)
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
    projectKey: Annotated[str, Field(description='Project key, e.g. "OCLOMRS"')],
    summary: Annotated[str, Field(description="One-line summary / title")],
    issueType: Annotated[
        str, Field(description='Issue type, e.g. "Bug", "Task", "Story"')
    ],
    description: Annotated[
        str | None, Field(description="Description in Markdown (converted to ADF)")
    ] = None,
    priority: Annotated[
        str | None, Field(description='Priority, e.g. "High", "Medium", "Low"')
    ] = None,
    labels: Annotated[list[str] | None, Field(description="Label strings")] = None,
    assigneeAccountId: Annotated[
        str | None, Field(description="Jira account ID for the assignee")
    ] = None,
) -> str:
    """Create a new Jira issue."""
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

    key = body.get("key") if isinstance(body, dict) else None
    if not key:
        raise ToolError(
            f"Jira accepted the request (HTTP {status}) but returned no issue key: "
            f"{body!r}"
        )
    return f"Created {key}: {JIRA_BASE_URL}/browse/{key}"


@mcp.tool()
async def getJiraIssue(
    issueIdOrKey: Annotated[
        str, Field(description='Issue key (e.g. "OCLOMRS-123") or numeric ID')
    ],
    fields: Annotated[
        list[str] | None,
        Field(description='Field names to return, e.g. ["summary", "status"]'),
    ] = None,
) -> str:
    """Fetch a Jira issue by key or ID."""
    params = {}
    if fields:
        params["fields"] = ",".join(fields)

    status, body = await _request(
        "GET", f"/rest/api/3/issue/{issueIdOrKey}", params=params
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    f = body.get("fields") if isinstance(body, dict) else None
    if not isinstance(f, dict):
        raise ToolError(
            f"Jira returned an unexpected issue shape (HTTP {status}): {body!r}"
        )
    key = body.get("key", "(unknown)")
    lines = [
        f"**{key}**: {f.get('summary', '(no summary)')}",
        f"URL: {JIRA_BASE_URL}/browse/{key}",
        f"Status: {f['status']['name']}" if f.get("status") else None,
        f"Type: {f['issuetype']['name']}" if f.get("issuetype") else None,
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
    issueIdOrKey: Annotated[
        str, Field(description='Issue key (e.g. "OCLOMRS-123") or numeric ID')
    ],
    summary: Annotated[str | None, Field(description="New summary / title")] = None,
    description: Annotated[
        str | None, Field(description="New description in Markdown (converted to ADF)")
    ] = None,
    priority: Annotated[
        str | None, Field(description='Priority, e.g. "High", "Medium", "Low"')
    ] = None,
    labels: Annotated[
        list[str] | None, Field(description="New set of labels (replaces existing)")
    ] = None,
    assigneeAccountId: Annotated[
        str | None, Field(description="Jira account ID for the assignee")
    ] = None,
    issueType: Annotated[
        str | None, Field(description='Issue type, e.g. "Bug", "Task", "Story"')
    ] = None,
) -> str:
    """Edit fields on an existing Jira issue."""
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
    jql: Annotated[
        str, Field(description='JQL query, e.g. "project = OCLOMRS AND status = Open"')
    ],
    maxResults: Annotated[
        int, Field(description="Max results (paginates automatically)")
    ] = 25,
    fields: Annotated[
        list[str] | None,
        Field(description='Fields to return, e.g. ["summary", "status"]'),
    ] = None,
) -> str:
    """Search for Jira issues using JQL."""
    issues: list[dict] = []
    next_page_token: str | None = None
    more_available = False

    while len(issues) < maxResults:
        page_size = min(_JIRA_PAGE_SIZE, maxResults - len(issues))
        params: dict = {
            "jql": jql,
            "maxResults": page_size,
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token
        if fields:
            params["fields"] = ",".join(fields)

        status, body = await _request("GET", "/rest/api/3/search/jql", params=params)
        if not _is_ok(status):
            raise ToolError(_error_message(status, body))
        if not isinstance(body, dict):
            raise ToolError(
                f"Jira returned an unexpected search response (HTTP {status}): {body!r}"
            )

        page = body.get("issues", [])
        issues.extend(page)

        next_page_token = body.get("nextPageToken")
        if body.get("isLast") is True or not next_page_token or not page:
            break
    else:
        # Loop exited because we hit maxResults; a remaining token means the
        # enhanced /search/jql endpoint (which does not report a total) has
        # more matches than we fetched.
        more_available = bool(next_page_token)

    header = f"Showing {len(issues)} issue(s)"
    if more_available:
        header += " (more available — increase maxResults)"
    lines = [f"{header}:"]
    for issue in issues:
        f = issue.get("fields", {}) if isinstance(issue, dict) else {}
        issue_status = f["status"]["name"] if f.get("status") else "?"
        assignee = f["assignee"]["displayName"] if f.get("assignee") else "Unassigned"
        summary = f.get("summary", "")
        key = issue.get("key", "(unknown)") if isinstance(issue, dict) else "(unknown)"
        lines.append(f"- **{key}** [{issue_status}] {summary} (Assignee: {assignee})")

    return "\n".join(lines)


@mcp.tool()
async def addCommentToJiraIssue(
    issueIdOrKey: Annotated[
        str, Field(description='Issue key (e.g. "OCLOMRS-123") or numeric ID')
    ],
    body: Annotated[
        str, Field(description="Comment text in Markdown (converted to ADF)")
    ],
) -> str:
    """Add a comment to a Jira issue."""
    status, resp_body = await _request(
        "POST",
        f"/rest/api/3/issue/{issueIdOrKey}/comment",
        json={"body": _markdown_to_adf(body)},
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, resp_body))

    comment_id = (
        resp_body.get("id", "(unknown)") if isinstance(resp_body, dict) else "(unknown)"
    )
    return f"Comment added (id: {comment_id}) to {issueIdOrKey}"


@mcp.tool()
async def getTransitionsForJiraIssue(
    issueIdOrKey: Annotated[
        str, Field(description='Issue key (e.g. "OCLOMRS-123") or numeric ID')
    ],
) -> str:
    """List available workflow transitions for a Jira issue."""
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
        to = t.get("to") if isinstance(t, dict) else None
        to_name = to.get("name", "?") if isinstance(to, dict) else "?"
        lines.append(f'- id={t.get("id")} name="{t.get("name")}" -> {to_name}')
    return "\n".join(lines)


@mcp.tool()
async def transitionJiraIssue(
    issueIdOrKey: Annotated[
        str, Field(description='Issue key (e.g. "OCLOMRS-123") or numeric ID')
    ],
    transitionId: Annotated[
        str, Field(description="Transition ID (from getTransitionsForJiraIssue)")
    ],
) -> str:
    """Transition a Jira issue to a new status."""
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
    if not isinstance(body, dict) or "accountId" not in body:
        raise ToolError(
            f"Jira returned an unexpected /myself response (HTTP {status}): {body!r}"
        )

    return (
        f"accountId: {body['accountId']}\n"
        f"displayName: {body.get('displayName', '(unknown)')}\n"
        f"emailAddress: {body.get('emailAddress', '(hidden)')}"
    )


@mcp.tool()
async def lookupJiraAccountId(
    query: Annotated[
        str,
        Field(description="Name or email address to search for"),
    ],
    maxResults: Annotated[int, Field(description="Max users to return")] = 10,
) -> str:
    """Search Jira users and return their account IDs."""
    status, body = await _request(
        "GET",
        "/rest/api/3/user/search",
        params={"query": query, "maxResults": maxResults},
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    users = body if isinstance(body, list) else []
    if not users:
        return f'No Jira users found for "{query}".'

    lines = [f"Found {len(users)} user(s):"]
    for user in users:
        display_name = user.get("displayName", "(unknown)")
        email = user.get("emailAddress", "hidden")
        account_id = user.get("accountId", "(unknown)")
        lines.append(f"- {display_name} <{email}> accountId={account_id}")
    return "\n".join(lines)


@mcp.tool()
async def getVisibleJiraProjects(
    maxResults: Annotated[int, Field(description="Max projects to return")] = 50,
) -> str:
    """List Jira projects visible to the authenticated user."""
    status, body = await _request(
        "GET", "/rest/api/3/project/search", params={"maxResults": maxResults}
    )
    if not _is_ok(status):
        raise ToolError(_error_message(status, body))

    projects = body.get("values", []) if isinstance(body, dict) else []
    lines = [f"Found {len(projects)} project(s):"]
    for p in projects:
        lines.append(f"- **{p.get('key', '?')}**: {p.get('name', '?')}")
    return "\n".join(lines)


@mcp.tool()
async def getJiraIssueTypeMetaWithFields(
    projectIdOrKey: Annotated[
        str, Field(description='Project key (e.g. "OCLOMRS") or numeric ID')
    ],
    issueTypeId: Annotated[str, Field(description="Numeric issue type ID")],
) -> str:
    """Get Jira create metadata for an issue type in a project."""
    fields_list: list[dict] = []
    start_at = 0
    total: int | None = None

    while True:
        page_size = _JIRA_PAGE_SIZE
        status, body = await _request(
            "GET",
            f"/rest/api/3/issue/createmeta/{projectIdOrKey}/issuetypes/{issueTypeId}",
            params={"startAt": start_at, "maxResults": page_size},
        )
        if not _is_ok(status):
            raise ToolError(_error_message(status, body))
        if not isinstance(body, dict):
            raise ToolError(
                f"Jira returned an unexpected createmeta response (HTTP {status}): "
                f"{body!r}"
            )

        page = body.get("values", body.get("fields", []))

        # Dict of field objects (older API shape)
        if isinstance(page, dict):
            lines = [f"Fields for project {projectIdOrKey}, issue type {issueTypeId}:"]
            for key, meta in page.items():
                required = " (required)" if meta.get("required") else ""
                lines.append(f"- {key}: {meta.get('name', key)}{required}")
            return "\n".join(lines)

        fields_list.extend(page)
        if "total" in body:
            total = body["total"]

        if body.get("isLast") is True or not page:
            break
        if total is not None and len(fields_list) >= total:
            break
        if total is None and len(page) < page_size:
            break
        start_at += len(page)

    # Paginated list shape
    lines = [f"Fields for project {projectIdOrKey}, issue type {issueTypeId}:"]
    for field in fields_list:
        required = " (required)" if field.get("required") else ""
        field_id = field.get("fieldId", field.get("key", "?"))
        name = field.get("name", field_id)
        lines.append(f"- {field_id}: {name}{required}")
    return "\n".join(lines)


# Upper bound on the reassembly buffer. A single JSON-RPC message should never
# approach this; exceeding it means the stream is unparseable garbage rather
# than a legitimately-large multiline message, so we resync instead of growing
# (and re-parsing) an ever-larger buffer forever.
_MAX_STDIN_BUFFER = 10 * 1024 * 1024


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

            if len(self._buffer) > _MAX_STDIN_BUFFER:
                print(
                    f"jira-mcp: stdin buffer exceeded {_MAX_STDIN_BUFFER} bytes "
                    "without forming valid JSON; discarding and resyncing",
                    file=sys.stderr,
                )
                self._buffer = ""

        # stdin exhausted. A leftover buffer that still does not parse is a
        # truncated/garbled final message — discard it rather than handing the
        # transport invalid JSON that would surface as an opaque parse error.
        if self._buffer:
            remaining, self._buffer = self._buffer, ""
            parsed = self._try_parse(remaining)
            if parsed is None:
                print(
                    f"jira-mcp: discarding {len(remaining)} bytes of unparseable "
                    "buffered stdin at EOF",
                    file=sys.stderr,
                )
                raise StopAsyncIteration
            return json.dumps(parsed, ensure_ascii=False) + "\n"
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
        try:
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
        finally:
            await _close_session()


def main():
    anyio.run(_run_stdio)


if __name__ == "__main__":
    main()
