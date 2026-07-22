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


async def test_request_connection_error_raises_toolerror():
    """Transport failures are surfaced as ToolError, not raw aiohttp errors."""
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/test",
            exception=aiohttp.ClientConnectionError("boom"),
        )
        with pytest.raises(ToolError, match="Could not reach Jira"):
            await main._request("GET", "/rest/api/3/test")


async def test_request_timeout_raises_toolerror():
    """A request timeout becomes a ToolError with actionable text."""
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/test",
            exception=TimeoutError(),
        )
        with pytest.raises(ToolError, match="timed out"):
            await main._request("GET", "/rest/api/3/test")


async def test_request_unparseable_json_raises_toolerror():
    """A JSON content-type with a non-JSON body is reported, not silently swallowed."""
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/test",
            body="<html>502 Bad Gateway</html>",
            content_type="application/json",
            status=502,
        )
        with pytest.raises(ToolError, match="unparseable response"):
            await main._request("GET", "/rest/api/3/test")


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


def test_error_message_with_list_body():
    """A JSON-array error body must not crash the formatter."""
    body = [{"some": "error"}]
    msg = main._error_message(400, body)
    assert json.dumps(body) in msg
    assert "HTTP 400" in msg


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


def test_markdown_to_adf_literal_asterisks_not_italic():
    """Whitespace-flanked asterisks (e.g. arithmetic) must stay literal text."""
    result = main._markdown_to_adf("2 * 3 * 4")
    nodes = result["content"][0]["content"]
    assert len(nodes) == 1
    assert nodes[0]["text"] == "2 * 3 * 4"
    assert "marks" not in nodes[0]


def test_markdown_to_adf_inline_italic_asterisk():
    """Genuine *italic* (markers abutting non-space) still converts."""
    result = main._markdown_to_adf("some *italic* text")
    nodes = result["content"][0]["content"]
    assert nodes[1]["text"] == "italic"
    assert nodes[1]["marks"] == [{"type": "em"}]


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


async def test_create_issue_with_parent():
    with aioresponses() as m:
        m.post(f"{JIRA_BASE}/rest/api/3/issue", payload={"key": "TEST-3"})
        await main.createJiraIssue(
            projectKey="TEST",
            summary="Child of an Epic",
            issueType="Task",
            parentKey="TEST-1",
        )

    assert m.requests is not None
    call = m.requests[("POST", yarl.URL(f"{JIRA_BASE}/rest/api/3/issue"))][0]
    assert call.kwargs["json"]["fields"]["parent"] == {"key": "TEST-1"}


async def test_create_issue_with_outward_link():
    with aioresponses() as m:
        m.post(f"{JIRA_BASE}/rest/api/3/issue", payload={"key": "TEST-5"})
        m.post(f"{JIRA_BASE}/rest/api/3/issueLink", status=201)
        result = await main.createJiraIssue(
            projectKey="TEST",
            summary="Blocker",
            issueType="Bug",
            links=[main.IssueLink(type="Blocks", issueKey="TEST-9")],
        )

    assert "Created TEST-5" in result
    assert "Linked TEST-5" in result

    assert m.requests is not None
    link_call = m.requests[("POST", yarl.URL(f"{JIRA_BASE}/rest/api/3/issueLink"))][0]
    payload = link_call.kwargs["json"]
    assert payload["type"] == {"name": "Blocks"}
    # Default direction "outward": the new issue blocks the other one.
    assert payload["outwardIssue"] == {"key": "TEST-5"}
    assert payload["inwardIssue"] == {"key": "TEST-9"}


async def test_create_issue_with_inward_link():
    with aioresponses() as m:
        m.post(f"{JIRA_BASE}/rest/api/3/issue", payload={"key": "TEST-7"})
        m.post(f"{JIRA_BASE}/rest/api/3/issueLink", status=201)
        await main.createJiraIssue(
            projectKey="TEST",
            summary="Blocked",
            issueType="Bug",
            links=[
                main.IssueLink(type="Blocks", issueKey="TEST-9", direction="inward")
            ],
        )

    assert m.requests is not None
    link_call = m.requests[("POST", yarl.URL(f"{JIRA_BASE}/rest/api/3/issueLink"))][0]
    payload = link_call.kwargs["json"]
    # Direction "inward": the new issue is blocked by the other one.
    assert payload["inwardIssue"] == {"key": "TEST-7"}
    assert payload["outwardIssue"] == {"key": "TEST-9"}


async def test_create_issue_reports_link_failure():
    """A link that fails after the issue is created is reported, not raised."""
    with aioresponses() as m:
        m.post(f"{JIRA_BASE}/rest/api/3/issue", payload={"key": "TEST-8"})
        m.post(
            f"{JIRA_BASE}/rest/api/3/issueLink",
            payload={"errorMessages": ["No issue link type with name 'Nope' found"]},
            status=404,
        )
        result = await main.createJiraIssue(
            projectKey="TEST",
            summary="s",
            issueType="Bug",
            links=[main.IssueLink(type="Nope", issueKey="TEST-9")],
        )

    assert "Created TEST-8" in result
    assert "FAILED to link" in result
    assert "No issue link type" in result


# ---------------------------------------------------------------------------
# editJiraIssue
# ---------------------------------------------------------------------------


async def test_edit_issue_fields():
    with aioresponses() as m:
        m.put(f"{JIRA_BASE}/rest/api/3/issue/TEST-1", status=204)
        result = await main.editJiraIssue("TEST-1", summary="New title")

    assert "Updated TEST-1" in result
    assert m.requests is not None
    call = m.requests[("PUT", yarl.URL(f"{JIRA_BASE}/rest/api/3/issue/TEST-1"))][0]
    assert call.kwargs["json"]["fields"]["summary"] == "New title"


async def test_edit_issue_sets_parent():
    with aioresponses() as m:
        m.put(f"{JIRA_BASE}/rest/api/3/issue/TEST-1", status=204)
        await main.editJiraIssue("TEST-1", parentKey="TEST-100")

    assert m.requests is not None
    call = m.requests[("PUT", yarl.URL(f"{JIRA_BASE}/rest/api/3/issue/TEST-1"))][0]
    assert call.kwargs["json"]["fields"]["parent"] == {"key": "TEST-100"}


async def test_edit_issue_clears_parent():
    """An empty parentKey clears the existing parent (sets the field to null)."""
    with aioresponses() as m:
        m.put(f"{JIRA_BASE}/rest/api/3/issue/TEST-1", status=204)
        await main.editJiraIssue("TEST-1", parentKey="")

    assert m.requests is not None
    call = m.requests[("PUT", yarl.URL(f"{JIRA_BASE}/rest/api/3/issue/TEST-1"))][0]
    assert call.kwargs["json"]["fields"]["parent"] is None


async def test_edit_issue_links_only_skips_put():
    """Editing only links makes no field-update PUT."""
    with aioresponses() as m:
        m.post(f"{JIRA_BASE}/rest/api/3/issueLink", status=201)
        result = await main.editJiraIssue(
            "TEST-1", links=[main.IssueLink(type="Relates", issueKey="TEST-2")]
        )

    assert "Linked TEST-1" in result
    assert m.requests is not None
    assert (
        "PUT",
        yarl.URL(f"{JIRA_BASE}/rest/api/3/issue/TEST-1"),
    ) not in m.requests


async def test_edit_issue_no_changes():
    result = await main.editJiraIssue("TEST-1")
    assert result == "No fields or links provided to update."


# ---------------------------------------------------------------------------
# linkJiraIssues / getJiraIssueLinkTypes
# ---------------------------------------------------------------------------


async def test_link_jira_issues():
    with aioresponses() as m:
        m.post(f"{JIRA_BASE}/rest/api/3/issueLink", status=201)
        result = await main.linkJiraIssues(
            linkType="Blocks", inwardIssue="TEST-2", outwardIssue="TEST-1"
        )

    assert "TEST-1 -> TEST-2" in result
    assert m.requests is not None
    call = m.requests[("POST", yarl.URL(f"{JIRA_BASE}/rest/api/3/issueLink"))][0]
    payload = call.kwargs["json"]
    assert payload["type"] == {"name": "Blocks"}
    assert payload["inwardIssue"] == {"key": "TEST-2"}
    assert payload["outwardIssue"] == {"key": "TEST-1"}


async def test_link_jira_issues_error():
    with aioresponses() as m:
        m.post(
            f"{JIRA_BASE}/rest/api/3/issueLink",
            payload={"errorMessages": ["Issue does not exist"]},
            status=404,
        )
        with pytest.raises(ToolError, match="Issue does not exist"):
            await main.linkJiraIssues(
                linkType="Blocks", inwardIssue="TEST-2", outwardIssue="TEST-1"
            )


async def test_get_issue_link_types():
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/issueLinkType",
            payload={
                "issueLinkTypes": [
                    {
                        "id": "1",
                        "name": "Blocks",
                        "inward": "is blocked by",
                        "outward": "blocks",
                    }
                ]
            },
        )
        result = await main.getJiraIssueLinkTypes()

    assert "Blocks" in result
    assert 'outward="blocks"' in result
    assert 'inward="is blocked by"' in result


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


async def test_get_issue_with_parent_and_links():
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/issue/TEST-1",
            payload={
                "key": "TEST-1",
                "fields": {
                    "summary": "Test issue",
                    "status": {"name": "Open"},
                    "parent": {
                        "key": "TEST-100",
                        "fields": {"summary": "The Epic"},
                    },
                    "issuelinks": [
                        {
                            "type": {
                                "name": "Blocks",
                                "inward": "is blocked by",
                                "outward": "blocks",
                            },
                            # This issue is the inward side (blocked by TEST-9).
                            "outwardIssue": {
                                "key": "TEST-9",
                                "fields": {
                                    "summary": "Blocker issue",
                                    "status": {"name": "In Progress"},
                                },
                            },
                        },
                        {
                            "type": {
                                "name": "Blocks",
                                "inward": "is blocked by",
                                "outward": "blocks",
                            },
                            # This issue is the outward side (blocks TEST-5).
                            "inwardIssue": {
                                "key": "TEST-5",
                                "fields": {
                                    "summary": "Blocked issue",
                                    "status": {"name": "Open"},
                                },
                            },
                        },
                    ],
                },
            },
        )
        result = await main.getJiraIssue("TEST-1")

    assert "Parent: TEST-100 The Epic" in result
    assert "Links:" in result
    assert "- is blocked by TEST-9 [In Progress] Blocker issue" in result
    assert "- blocks TEST-5 [Open] Blocked issue" in result


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

    assert "Showing 1 issue(s)" in result
    assert "TEST-1" in result
    assert "Unassigned" in result


async def test_search_issues_reports_more_available(monkeypatch):
    """Hitting maxResults with a nextPageToken left over flags more matches."""
    monkeypatch.setattr(main, "_JIRA_PAGE_SIZE", 2)

    search_url = stdlib_re.compile(
        r"^https://openmrs\.atlassian\.net/rest/api/3/search/jql"
    )
    with aioresponses() as m:
        m.get(
            search_url,
            payload={
                "isLast": False,
                "nextPageToken": "page-2",
                "issues": [
                    {
                        "key": "TEST-1",
                        "fields": {
                            "summary": "One",
                            "status": {"name": "Open"},
                            "assignee": None,
                        },
                    },
                    {
                        "key": "TEST-2",
                        "fields": {
                            "summary": "Two",
                            "status": {"name": "Open"},
                            "assignee": None,
                        },
                    },
                ],
            },
        )
        result = await main.searchJiraIssues(jql="project = TEST", maxResults=2)

    assert "Showing 2 issue(s)" in result
    assert "more available" in result


async def test_search_issues_paginates(monkeypatch):
    """When Jira returns nextPageToken, it is used for the next page."""
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
                "isLast": False,
                "nextPageToken": "page-2",
                "issues": [_make_issue("TEST-1"), _make_issue("TEST-2")],
            },
        )
        # Page 2: 1 issue (last page)
        m.get(
            search_url,
            payload={
                "isLast": True,
                "issues": [_make_issue("TEST-3")],
            },
        )
        result = await main.searchJiraIssues(jql="project = TEST", maxResults=5)

    assert "Showing 3 issue(s)" in result
    assert "TEST-1" in result
    assert "TEST-2" in result
    assert "TEST-3" in result
    assert m.requests is not None
    calls = [call for calls in m.requests.values() for call in calls]
    assert calls[0].kwargs["params"] == {
        "jql": "project = TEST",
        "maxResults": 2,
    }
    assert calls[1].kwargs["params"] == {
        "jql": "project = TEST",
        "maxResults": 2,
        "nextPageToken": "page-2",
    }


# ---------------------------------------------------------------------------
# getJiraIssueTypeMetaWithFields
# ---------------------------------------------------------------------------


async def test_issue_type_meta_paginates_fields(monkeypatch):
    monkeypatch.setattr(main, "_JIRA_PAGE_SIZE", 1)

    meta_url = stdlib_re.compile(
        r"^https://openmrs\.atlassian\.net/rest/api/3/issue/createmeta/TEST/issuetypes/10001"
    )
    with aioresponses() as m:
        m.get(
            meta_url,
            payload={
                "fields": [{"fieldId": "summary", "name": "Summary", "required": True}],
                "maxResults": 1,
                "startAt": 0,
                "total": 2,
            },
        )
        m.get(
            meta_url,
            payload={
                "fields": [
                    {
                        "fieldId": "description",
                        "name": "Description",
                        "required": False,
                    }
                ],
                "maxResults": 1,
                "startAt": 1,
                "total": 2,
            },
        )
        result = await main.getJiraIssueTypeMetaWithFields("TEST", "10001")

    assert "- summary: Summary (required)" in result
    assert "- description: Description" in result
    assert m.requests is not None
    calls = [call for calls in m.requests.values() for call in calls]
    assert calls[0].kwargs["params"] == {"startAt": 0, "maxResults": 1}
    assert calls[1].kwargs["params"] == {"startAt": 1, "maxResults": 1}


# ---------------------------------------------------------------------------
# lookupJiraAccountId
# ---------------------------------------------------------------------------


async def test_lookup_jira_account_id():
    user_search_url = stdlib_re.compile(
        r"^https://openmrs\.atlassian\.net/rest/api/3/user/search"
    )
    with aioresponses() as m:
        m.get(
            user_search_url,
            payload=[
                {
                    "accountId": "abc123",
                    "displayName": "Alice Example",
                    "emailAddress": "alice@example.com",
                },
                {
                    "accountId": "def456",
                    "displayName": "Bob Example",
                },
            ],
        )
        result = await main.lookupJiraAccountId(query="alice", maxResults=10)

    assert "Found 2 user(s):" in result
    assert "- Alice Example <alice@example.com> accountId=abc123" in result
    assert "- Bob Example <hidden> accountId=def456" in result
    assert m.requests is not None
    call = next(call for calls in m.requests.values() for call in calls)
    assert call.kwargs["params"] == {"query": "alice", "maxResults": 10}


# ---------------------------------------------------------------------------
# getMyself
# ---------------------------------------------------------------------------


async def test_get_myself():
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/myself",
            payload={
                "accountId": "abc123",
                "displayName": "Alice",
                "emailAddress": "alice@example.com",
            },
        )
        result = await main.getMyself()

    assert "accountId: abc123" in result
    assert "displayName: Alice" in result
    assert "emailAddress: alice@example.com" in result


async def test_get_myself_api_error():
    with aioresponses() as m:
        m.get(
            f"{JIRA_BASE}/rest/api/3/myself",
            payload={"errorMessages": ["Not authenticated"]},
            status=401,
        )
        with pytest.raises(ToolError, match="Not authenticated"):
            await main.getMyself()


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


async def test_buffered_stdin_discards_unparseable_at_eof():
    """A garbled final message that never parses is discarded, not forwarded."""
    fake = _FakeAsyncStdin(['{"jsonrpc": "2.0", "method": "trunc\n'])
    buffered = _JsonLineBufferedStdin(fake)
    results = await _collect(buffered)
    assert results == []


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
