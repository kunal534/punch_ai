from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, AnyUrl, Field
import readabilipy
from pathlib import Path
import re

# -----------------------------
# Configuration
# -----------------------------
TOKEN = "a73bd1180050"  # Replace with your application key
MY_NUMBER = "916396810512"  # Replace with your phone number {country_code}{number}

# -----------------------------
# Tool Description Model
# -----------------------------
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None

# -----------------------------
# Authentication Provider
# -----------------------------
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="unknown", scopes=[], expires_at=None)
        return None

# -----------------------------
# Fetch Tool
# -----------------------------
class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(cls, url: str, user_agent: str, force_raw: bool = False) -> tuple[str, str]:
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(url, follow_redirects=True, headers={"User-Agent": user_agent}, timeout=30)
            except HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "<html" in page_raw[:100] or "text/html" in content_type or not content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (page_raw, f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n")

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

# -----------------------------
# FastMCP Server
# -----------------------------
mcp = FastMCP("My MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

# -----------------------------
# Resume Tool
# -----------------------------
ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; must return raw markdown, no extra formatting.",
    side_effects=None,
)

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Reads resume.pdf, converts to simple markdown, and returns it.
    """
    resume_path = Path("resume.pdf")
    if not resume_path.exists():
        return "<error>resume.pdf not found in the server directory.</error>"

    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(resume_path))
        if not text.strip():
            return "<error>Could not extract text from resume.pdf.</error>"

        # Simple markdown formatting for common headers
        text = re.sub(r"(Education|Experience|Projects|Skills|Certifications)", r"## \1", text)
        # Convert numbered lists to markdown bullets
        text = re.sub(r"^\d+\.\s", r"- ", text, flags=re.MULTILINE)

        return text
    except Exception as e:
        return f"<error>Failed to process resume.pdf: {e}</error>"

# -----------------------------
# Validate Tool
# -----------------------------
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# -----------------------------
# Fetch Tool
# -----------------------------
FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="User provides a URL and asks for content.",
    side_effects="Returns simplified markdown or raw HTML if requested.",
)

@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[int, Field(default=5000, description="Max chars", gt=0, lt=1000000)] = 5000,
    start_index: Annotated[int, Field(default=0, description="Start index for truncated fetch", ge=0)] = 0,
    raw: Annotated[bool, Field(default=False, description="Return raw HTML")] = False,
) -> list[TextContent]:
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_length = len(truncated_content)
            remaining = original_length - (start_index + actual_length)
            if actual_length == max_length and remaining > 0:
                next_start = start_index + actual_length
                content += f"\n\n<error>Content truncated. Call fetch with start_index={next_start} to continue.</error>"

    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]

@mcp.tool
async def about() -> dict:
    return {
        "name": "My MCP Server",
        "description": "Server hosting tools to share my resume and fetch web content"
    }
# -----------------------------
# Run Server
# -----------------------------
async def main():
    await mcp.run_async(
        transport="streamable-http",
        host="0.0.0.0",
        port=8085
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
