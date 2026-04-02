# jira-mcp

A local MCP server that wraps the Jira REST API v3 for `openmrs.atlassian.net`, providing a comprehensive alternative to the Atlassian Rovo MCP Server. Bypasses the Rovo server's "anonymous project" bug by calling the REST API directly with Basic Auth.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- A Jira API token (see below)

## Generating a Jira API Token

1. Go to <https://id.atlassian.com/manage-profile/security/api-tokens>
2. Click **Create API token**, give it a label (e.g. "Claude MCP"), and copy the token.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `JIRA_EMAIL` | Yes | Your Atlassian account email |
| `JIRA_API_TOKEN` | Yes | The API token you generated above |
| `JIRA_BASE_URL` | No | Defaults to `https://openmrs.atlassian.net` |

## Claude Desktop MCP Config

Add the following to your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux / WSL**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "jira": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/jira-mcp", "python", "main.py"],
      "env": {
        "JIRA_EMAIL": "you@example.com",
        "JIRA_API_TOKEN": "your-api-token-here"
      }
    }
  }
}
```

Replace `/absolute/path/to/jira-mcp` with the actual path to this directory.

## Claude Code MCP Config

In your Claude Code settings (e.g. `~/.claude/settings.json`), add:

```json
{
  "mcpServers": {
    "jira": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/jira-mcp", "python", "main.py"],
      "env": {
        "JIRA_EMAIL": "you@example.com",
        "JIRA_API_TOKEN": "your-api-token-here"
      }
    }
  }
}
```

## Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `createJiraIssue` | Create a new issue | `projectKey`, `summary`, `issueType`, `description`, `priority`, `labels`, `assigneeAccountId` |
| `getJiraIssue` | Fetch an issue by key/ID | `issueIdOrKey`, `fields` |
| `editJiraIssue` | Update fields on an issue | `issueIdOrKey`, `summary`, `description`, `priority`, `labels`, `assigneeAccountId`, `issueType` |
| `searchJiraIssues` | Search with JQL | `jql`, `maxResults`, `fields` |
| `addCommentToJiraIssue` | Add a comment | `issueIdOrKey`, `body` |
| `getTransitionsForJiraIssue` | List available transitions | `issueIdOrKey` |
| `transitionJiraIssue` | Move issue to new status | `issueIdOrKey`, `transitionId` |
| `getVisibleJiraProjects` | List visible projects | `maxResults` |
| `getJiraIssueTypeMetaWithFields` | Get create metadata for a project/issue type | `projectIdOrKey`, `issueTypeId` |
| `lookupJiraAccountId` | Search users by name/email | `query`, `maxResults` |

All `description` and comment `body` parameters accept Markdown, which is automatically converted to Atlassian Document Format (ADF) before sending.
