import json
import re as stdlib_re
from base64 import b64encode

import aiohttp
import pytest
import yarl
from aioresponses import aioresponses
from mcp.server.fastmcp.exceptions import ToolError

import main
from main import _JsonLineBufferedStdin

JIRA_BASE = "https://openmrs.atlassian.net"


@pytest.fixture(autouse=True)
def _jira_env(monkeypatch):
    monkeypatch.setenv("JIRA_EMAIL", "test@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "fake-token")


@pytest.fixture(autouse=True)
async def _reset_session():
    """Ensure each test starts with a fresh session."""
    main._session = None
    yield
    if main._session and not main._session.closed:
        await main._session.close()
    main._session = None


# ---------------------------------------------------------------------------
# _get_session / session reuse
# ---------------------------------------------------------------------------


async def test_get_session_creates_session():
    session = main._get_session()
    assert isinstance(session, aiohttp.ClientSession)


async def test_get_session_reuses_session():
    s1 = main._get_session()
    s2 = main._get_session()
    assert s1 is s2


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------


async def test_request_sends_auth_header():
    expected_cred = b64encode(b"test@example.com:fake-token").decode()
    with aioresponses() as m:
        m.get(f"{JIRA_BASE}/rest/api/3/test", payload={"ok": True})
        status, body = await main._request("GET", "/rest/api/3/test")

    assert status == 200
    assert body == {"ok": True}
    assert m.requests is not None
    call = m.requests[("GET", yarl.URL(f"{JIRA_BASE}/rest/api/3/test"))][0]
    assert call.kwargs["headers"]["Authorization"] == f"Basic {expected_cred}"


async def test_request_returns_text_for_non_json():
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/test",
            body="plain text",
            content_type="text/plain",
        )
        status, body = await main._request("GET", "/rest/api/3/test")

    assert status == 200
    assert body == "plain text"


# ---------------------------------------------------------------------------
# _error_message
# ---------------------------------------------------------------------------


def test_error_message_with_string_body():
    msg = main._error_message(500, "Server error")
    assert msg == "Jira error (HTTP 500): Server error"


def test_error_message_with_error_messages():
    body = {"errorMessages": ["Issue not found", "Permission denied"]}
    msg = main._error_message(404, body)
    assert "Issue not found" in msg
    assert "Permission denied" in msg


def test_error_message_with_field_errors():
    body = {"errors": {"summary": "Field is required"}}
    msg = main._error_message(400, body)
    assert "summary: Field is required" in msg


def test_error_message_fallback_to_json():
    body = {"unexpected": "structure"}
    msg = main._error_message(400, body)
    assert json.dumps(body) in msg


# ---------------------------------------------------------------------------
# _markdown_to_adf
# ---------------------------------------------------------------------------


def test_markdown_to_adf_plain_paragraph():
    result = main._markdown_to_adf("Hello world")
    assert result["type"] == "doc"
    assert result["version"] == 1
    assert len(result["content"]) == 1
    para = result["content"][0]
    assert para["type"] == "paragraph"
    assert para["content"][0]["text"] == "Hello world"


def test_markdown_to_adf_heading():
    result = main._markdown_to_adf("## Section Title")
    heading = result["content"][0]
    assert heading["type"] == "heading"
    assert heading["attrs"]["level"] == 2
    assert heading["content"][0]["text"] == "Section Title"


def test_markdown_to_adf_bullet_list():
    md = "- item one\n- item two"
    result = main._markdown_to_adf(md)
    bl = result["content"][0]
    assert bl["type"] == "bulletList"
    assert len(bl["content"]) == 2


def test_markdown_to_adf_ordered_list():
    md = "1. first\n2. second\n3. third"
    result = main._markdown_to_adf(md)
    ol = result["content"][0]
    assert ol["type"] == "orderedList"
    assert len(ol["content"]) == 3


def test_markdown_to_adf_code_block():
    md = "```python\nprint('hi')\n```"
    result = main._markdown_to_adf(md)
    cb = result["content"][0]
    assert cb["type"] == "codeBlock"
    assert cb["attrs"]["language"] == "python"
    assert cb["content"][0]["text"] == "print('hi')"


def test_markdown_to_adf_inline_bold():
    result = main._markdown_to_adf("some **bold** text")
    nodes = result["content"][0]["content"]
    assert nodes[0]["text"] == "some "
    assert nodes[1]["text"] == "bold"
    assert nodes[1]["marks"] == [{"type": "strong"}]


def test_markdown_to_adf_italic_underscore():
    result = main._markdown_to_adf("some _italic_ text")
    nodes = result["content"][0]["content"]
    assert nodes[1]["text"] == "italic"
    assert nodes[1]["marks"] == [{"type": "em"}]


def test_markdown_to_adf_underscore_in_identifier():
    """Mid-word underscores must not be treated as italic markers."""
    result = main._markdown_to_adf("the release_publish_command option")
    nodes = result["content"][0]["content"]
    assert len(nodes) == 1
    assert nodes[0]["text"] == "the release_publish_command option"


def test_markdown_to_adf_inline_code():
    result = main._markdown_to_adf("use `foo()` here")
    nodes = result["content"][0]["content"]
    assert nodes[1]["text"] == "foo()"
    assert nodes[1]["marks"] == [{"type": "code"}]


def test_markdown_to_adf_literal_backslash_n():
    """Literal \\n escape sequences are normalised to real newlines before parsing."""
    md = (
        "First paragraph.\\n\\nSecond paragraph."
        "\\n\\n```yaml\\nif: true\\n```"
        "\\n\\nThe `pre-release` job."
    )
    result = main._markdown_to_adf(md)
    types = [node["type"] for node in result["content"]]
    assert types == ["paragraph", "paragraph", "codeBlock", "paragraph"]

    # Code block has the right language and content
    cb = result["content"][2]
    assert cb["attrs"]["language"] == "yaml"
    assert cb["content"][0]["text"] == "if: true"

    # Inline code in the last paragraph is preserved
    last_para = result["content"][3]
    code_nodes = [n for n in last_para["content"] if n.get("marks")]
    assert code_nodes[0]["text"] == "pre-release"
    assert code_nodes[0]["marks"] == [{"type": "code"}]


def test_markdown_to_adf_real_newlines():
    """Real newlines still work after the \\n normalisation."""
    md = (
        "First paragraph.\n\nSecond paragraph."
        "\n\n```yaml\nif: true\n```"
        "\n\nThe `pre-release` job."
    )
    result = main._markdown_to_adf(md)
    types = [node["type"] for node in result["content"]]
    assert types == ["paragraph", "paragraph", "codeBlock", "paragraph"]

    cb = result["content"][2]
    assert cb["attrs"]["language"] == "yaml"
    assert cb["content"][0]["text"] == "if: true"

    last_para = result["content"][3]
    code_nodes = [n for n in last_para["content"] if n.get("marks")]
    assert code_nodes[0]["text"] == "pre-release"
    assert code_nodes[0]["marks"] == [{"type": "code"}]


def test_markdown_to_adf_link():
    result = main._markdown_to_adf("see [OpenMRS](https://openmrs.org)")
    nodes = result["content"][0]["content"]
    link_node = nodes[1]
    assert link_node["text"] == "OpenMRS"
    assert link_node["marks"][0]["type"] == "link"
    assert link_node["marks"][0]["attrs"]["href"] == "https://openmrs.org"


# ---------------------------------------------------------------------------
# createJiraIssue
# ---------------------------------------------------------------------------


async def test_create_issue_minimal():
    with aioresponses() as m:
        m.post(
            f"{JIRA_BASE}/rest/api/3/issue",
            payload={"id": "10001", "key": "TEST-1", "self": "..."},
        )
        result = await main.createJiraIssue(
            projectKey="TEST",
            summary="A test issue",
            issueType="Bug",
        )

    assert "Created TEST-1" in result
    assert f"{JIRA_BASE}/browse/TEST-1" in result

    assert m.requests is not None
    call = m.requests[("POST", yarl.URL(f"{JIRA_BASE}/rest/api/3/issue"))][0]
    sent = call.kwargs["json"]
    assert sent["fields"]["project"] == {"key": "TEST"}
    assert sent["fields"]["summary"] == "A test issue"
    assert sent["fields"]["issuetype"] == {"name": "Bug"}
    assert "description" not in sent["fields"]
    assert "priority" not in sent["fields"]
    assert "labels" not in sent["fields"]
    assert "assignee" not in sent["fields"]


async def test_create_issue_all_fields():
    with aioresponses() as m:
        m.post(
            f"{JIRA_BASE}/rest/api/3/issue",
            payload={"id": "10002", "key": "TEST-2", "self": "..."},
        )
        result = await main.createJiraIssue(
            projectKey="TEST",
            summary="Full issue",
            issueType="Story",
            description="Some **bold** description",
            priority="High",
            labels=["backend", "urgent"],
            assigneeAccountId="abc123",
        )

    assert "Created TEST-2" in result

    assert m.requests is not None
    call = m.requests[("POST", yarl.URL(f"{JIRA_BASE}/rest/api/3/issue"))][0]
    fields = call.kwargs["json"]["fields"]
    assert fields["priority"] == {"name": "High"}
    assert fields["labels"] == ["backend", "urgent"]
    assert fields["assignee"] == {"accountId": "abc123"}
    assert fields["description"]["type"] == "doc"
    assert fields["description"]["version"] == 1


async def test_create_issue_api_error():
    with aioresponses() as m:
        m.post(
            f"{JIRA_BASE}/rest/api/3/issue",
            payload={"errorMessages": [], "errors": {"summary": "Field is required"}},
            status=400,
        )
        with pytest.raises(ToolError, match="summary"):
            await main.createJiraIssue(
                projectKey="TEST",
                summary="",
                issueType="Bug",
            )


# ---------------------------------------------------------------------------
# getJiraIssue
# ---------------------------------------------------------------------------


async def test_get_issue():
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/issue/TEST-1",
            payload={
                "key": "TEST-1",
                "fields": {
                    "summary": "Test issue",
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": "High"},
                    "assignee": {"displayName": "Alice"},
                    "labels": ["backend"],
                    "description": None,
                },
            },
        )
        result = await main.getJiraIssue("TEST-1")

    assert "TEST-1" in result
    assert "Test issue" in result
    assert "Open" in result
    assert "Bug" in result
    assert "High" in result
    assert "Alice" in result
    assert "backend" in result


# ---------------------------------------------------------------------------
# searchJiraIssues
# ---------------------------------------------------------------------------


async def test_search_issues():
    with aioresponses() as m:
        m.get(
            stdlib_re.compile(
                r"^https://openmrs\.atlassian\.net/rest/api/3/search/jql"
            ),
            payload={
                "total": 1,
                "issues": [
                    {
                        "key": "TEST-1",
                        "fields": {
                            "summary": "Found issue",
                            "status": {"name": "Open"},
                            "assignee": None,
                        },
                    }
                ],
            },
        )
        result = await main.searchJiraIssues(jql="project = TEST")

    assert "Found 1 issue(s)" in result
    assert "TEST-1" in result
    assert "Unassigned" in result


async def test_search_issues_paginates(monkeypatch):
    """When maxResults exceeds the page size, multiple requests are made."""
    monkeypatch.setattr(main, "_JIRA_PAGE_SIZE", 2)

    def _make_issue(key):
        return {
            "key": key,
            "fields": {
                "summary": f"Issue {key}",
                "status": {"name": "Open"},
                "assignee": None,
            },
        }

    search_url = stdlib_re.compile(
        r"^https://openmrs\.atlassian\.net/rest/api/3/search/jql"
    )
    with aioresponses() as m:
        # Page 1: 2 issues (full page)
        m.get(
            search_url,
            payload={
                "total": 3,
                "issues": [_make_issue("TEST-1"), _make_issue("TEST-2")],
            },
        )
        # Page 2: 1 issue (last page)
        m.get(
            search_url,
            payload={
                "total": 3,
                "issues": [_make_issue("TEST-3")],
            },
        )
        result = await main.searchJiraIssues(jql="project = TEST", maxResults=5)

    assert "Found 3 issue(s) (showing 3)" in result
    assert "TEST-1" in result
    assert "TEST-2" in result
    assert "TEST-3" in result


# ---------------------------------------------------------------------------
# _JsonLineBufferedStdin
# ---------------------------------------------------------------------------


class _FakeAsyncStdin:
    """Simulate an async stdin that yields pre-defined lines."""

    def __init__(self, lines: list[str]):
        self._iter = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


async def _collect(buffered: _JsonLineBufferedStdin) -> list[str]:
    results = []
    async for line in buffered:
        results.append(line)
    return results


async def test_buffered_stdin_single_line_json():
    """A well-formed single-line JSON message passes through immediately."""
    msg = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
    fake = _FakeAsyncStdin([msg + "\n"])
    buffered = _JsonLineBufferedStdin(fake)
    results = await _collect(buffered)
    assert len(results) == 1
    assert json.loads(results[0]) == {"jsonrpc": "2.0", "method": "test", "id": 1}


async def test_buffered_stdin_multiline_string():
    """Literal newlines in a JSON string are reassembled."""
    # Simulate what Claude Desktop sends: a JSON object where a string value
    # contains literal newlines instead of \\n escapes.
    original = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "createJiraIssue",
            "arguments": {"description": "line one\nline two\nline three"},
        },
        "id": 1,
    }
    # The properly-escaped JSON on one line:
    proper_json = json.dumps(original)

    # Now simulate what the buggy client sends: literal newlines in the string.
    # This means the single JSON line becomes 3 lines on the wire.
    broken = proper_json.replace("\\n", "\n")
    lines = [part + "\n" for part in broken.split("\n")]

    fake = _FakeAsyncStdin(lines)
    buffered = _JsonLineBufferedStdin(fake)
    results = await _collect(buffered)
    assert len(results) == 1
    parsed = json.loads(results[0])
    assert (
        parsed["params"]["arguments"]["description"] == "line one\nline two\nline three"
    )


async def test_buffered_stdin_multiple_messages():
    """Multiple well-formed messages are each yielded individually."""
    msg1 = json.dumps({"jsonrpc": "2.0", "method": "a", "id": 1}) + "\n"
    msg2 = json.dumps({"jsonrpc": "2.0", "method": "b", "id": 2}) + "\n"
    fake = _FakeAsyncStdin([msg1, msg2])
    buffered = _JsonLineBufferedStdin(fake)
    results = await _collect(buffered)
    assert len(results) == 2
    assert json.loads(results[0])["method"] == "a"
    assert json.loads(results[1])["method"] == "b"


async def test_buffered_stdin_mixed_good_and_broken():
    """A well-formed message followed by a broken multiline one both come through."""
    good = json.dumps({"jsonrpc": "2.0", "method": "init", "id": 1}) + "\n"

    broken_obj = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "arguments": {"body": "## Heading\n\nParagraph"},
        },
        "id": 2,
    }
    broken_json = json.dumps(broken_obj).replace("\\n", "\n")
    broken_lines = [part + "\n" for part in broken_json.split("\n")]

    fake = _FakeAsyncStdin([good] + broken_lines)
    buffered = _JsonLineBufferedStdin(fake)
    results = await _collect(buffered)
    assert len(results) == 2
    assert json.loads(results[0])["method"] == "init"
    assert (
        json.loads(results[1])["params"]["arguments"]["body"]
        == "## Heading\n\nParagraph"
    )
