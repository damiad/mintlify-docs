from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol

from check_api_docs import EXPECTED_SKILLS

DOCS_ORIGIN = os.getenv("FASTINO_DOCS_ORIGIN", "https://docs.fastino.ai").rstrip("/")
API_ORIGIN = os.getenv("FASTINO_API_ORIGIN", "https://api.fastino.ai").rstrip("/")
AGENT_ORIGIN = os.getenv("FASTINO_AGENT_ORIGIN", "https://agent.fastino.ai").rstrip("/")
MARKETING_ORIGIN = os.getenv("FASTINO_MARKETING_ORIGIN", "https://fastino.ai").rstrip(
    "/"
)
USER_AGENT = "FastinoDocsCanary/1.0 (+https://docs.fastino.ai)"
GOOGLEBOT_USER_AGENT = (
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
)
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DiagnosticsFailure(RuntimeError):
    pass


class HttpResponse(Protocol):
    status: int
    url: str
    headers: dict[str, str]
    body: bytes

    def text(self) -> str:
        ...

    def json(self) -> object:
        ...


class Request(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: object | None = None,
        follow_redirects: bool = True,
        timeout: float = 30,
    ) -> HttpResponse:
        ...


OpenApiValidator = Callable[[dict[str, object]], object]


@dataclass(frozen=True)
class DiscoveryTarget:
    name: str
    url: str
    content_types: tuple[str, ...]
    max_cache_seconds: int
    required_fragments: tuple[str, ...] = ()
    allow_noindex: bool = False


@dataclass(frozen=True)
class SkillEntry:
    name: str
    url: str
    digest: str


def base_discovery_targets() -> tuple[DiscoveryTarget, ...]:
    return (
        DiscoveryTarget(
            "agent sitemap",
            f"{AGENT_ORIGIN}/sitemap.xml",
            ("application/xml", "text/xml"),
            3600,
            (f"{AGENT_ORIGIN}/llms.txt",),
            True,
        ),
        DiscoveryTarget(
            "agent robots",
            f"{AGENT_ORIGIN}/robots.txt",
            ("text/plain",),
            3600,
            ("User-agent: Googlebot", "Allow: /"),
            True,
        ),
        DiscoveryTarget(
            "agent card",
            f"{AGENT_ORIGIN}/.well-known/agent-card.json",
            ("application/json",),
            300,
            allow_noindex=True,
        ),
        DiscoveryTarget(
            "agent skill index",
            f"{AGENT_ORIGIN}/.well-known/agent-skills/index.json",
            ("application/json",),
            3600,
            tuple(sorted(EXPECTED_SKILLS)),
            True,
        ),
        DiscoveryTarget(
            "canonical skill index",
            f"{DOCS_ORIGIN}/.well-known/agent-skills/index.json",
            ("application/json",),
            3600,
            tuple(sorted(EXPECTED_SKILLS)),
        ),
        DiscoveryTarget(
            "agent llms",
            f"{AGENT_ORIGIN}/llms.txt",
            ("text/markdown", "text/plain"),
            3600,
            ("Fastino",),
            True,
        ),
        DiscoveryTarget(
            "canonical llms",
            f"{DOCS_ORIGIN}/llms.txt",
            ("text/markdown", "text/plain"),
            3600,
            ("Fastino",),
        ),
        DiscoveryTarget(
            "API sitemap",
            f"{API_ORIGIN}/sitemap.xml",
            ("application/xml", "text/xml"),
            3600,
            (f"{DOCS_ORIGIN}/.well-known/agent-skills/index.json",),
            True,
        ),
        DiscoveryTarget(
            "API robots",
            f"{API_ORIGIN}/robots.txt",
            ("text/plain",),
            3600,
            ("Allow: /.well-known/", "search=yes"),
            True,
        ),
        DiscoveryTarget(
            "API catalog",
            f"{API_ORIGIN}/.well-known/api-catalog",
            ("application/json",),
            3600,
            (f"{DOCS_ORIGIN}/openapi.json",),
        ),
        DiscoveryTarget(
            "API agent card",
            f"{API_ORIGIN}/.well-known/agent-card.json",
            ("application/json",),
            3600,
        ),
        DiscoveryTarget(
            "canonical OpenAPI",
            f"{DOCS_ORIGIN}/openapi.json",
            ("application/json", "application/openapi+json"),
            3600,
        ),
        DiscoveryTarget(
            "marketing API catalog",
            f"{MARKETING_ORIGIN}/.well-known/api-catalog",
            ("application/json",),
            86400,
            ("openapi",),
        ),
    )


def _header(response: HttpResponse, name: str) -> str:
    return response.headers.get(name.lower(), "")


def _cache_directives(value: str) -> dict[str, str | None]:
    directives: dict[str, str | None] = {}
    for raw_directive in value.split(","):
        name, separator, raw_value = raw_directive.strip().partition("=")
        if name:
            directives[name.lower()] = (
                raw_value.strip().strip('"') if separator else None
            )
    return directives


def _cache_seconds(
    directives: dict[str, str | None],
    name: str,
    *,
    target_name: str,
    findings: list[str],
) -> int | None:
    value = directives.get(name)
    if value is None:
        return None
    try:
        seconds = int(value)
    except ValueError:
        findings.append(f"{target_name}: invalid {name} cache directive: {value!r}")
        return None
    if seconds < 0:
        findings.append(f"{target_name}: invalid {name} cache directive: {value!r}")
        return None
    return seconds


def discovery_response_findings(
    target: DiscoveryTarget,
    response: HttpResponse,
) -> list[str]:
    findings: list[str] = []
    if response.status != 200:
        findings.append(
            f"{target.name}: public discovery returned HTTP {response.status}"
        )

    content_type = _header(response, "content-type").partition(";")[0].strip().lower()
    if content_type not in target.content_types:
        findings.append(
            f"{target.name}: unexpected Content-Type {content_type or '<missing>'}"
        )

    cache_control = _header(response, "cache-control")
    if not cache_control:
        findings.append(f"{target.name}: missing Cache-Control")
    directives = _cache_directives(cache_control)
    robots_directives = {
        directive.strip().lower()
        for directive in _header(response, "x-robots-tag").split(",")
    }
    if not target.allow_noindex and "noindex" in robots_directives:
        findings.append(f"{target.name}: X-Robots-Tag blocks indexing with noindex")

    max_age = _cache_seconds(
        directives,
        "s-maxage",
        target_name=target.name,
        findings=findings,
    )
    if max_age is None and "s-maxage" not in directives:
        max_age = _cache_seconds(
            directives,
            "max-age",
            target_name=target.name,
            findings=findings,
        )
    explicitly_revalidated = (
        "no-cache" in directives
        or "no-store" in directives
        or (max_age == 0 and "must-revalidate" in directives)
    )
    if (
        max_age is None
        and "s-maxage" not in directives
        and "max-age" not in directives
        and not explicitly_revalidated
    ):
        findings.append(f"{target.name}: missing cache freshness directive")
    if max_age is not None and max_age > target.max_cache_seconds:
        findings.append(
            f"{target.name}: cache lifetime {max_age}s exceeds "
            f"{target.max_cache_seconds}s"
        )

    stale_window = _cache_seconds(
        directives,
        "stale-while-revalidate",
        target_name=target.name,
        findings=findings,
    )
    if stale_window is not None and stale_window > target.max_cache_seconds:
        findings.append(
            f"{target.name}: stale-while-revalidate={stale_window} exceeds "
            f"{target.max_cache_seconds}s"
        )
    if (
        max_age is not None
        and stale_window is not None
        and max_age + stale_window > target.max_cache_seconds
        and max_age <= target.max_cache_seconds
        and stale_window <= target.max_cache_seconds
    ):
        findings.append(
            f"{target.name}: cache lifetime plus stale window "
            f"{max_age + stale_window}s exceeds {target.max_cache_seconds}s"
        )

    age_header = _header(response, "age")
    if age_header:
        try:
            age = int(age_header)
        except ValueError:
            findings.append(f"{target.name}: invalid Age header {age_header!r}")
        else:
            stale_deadline = (
                max_age + (stale_window or 0) if max_age is not None else None
            )
            if (
                stale_deadline is not None
                and age > stale_deadline
                and not explicitly_revalidated
            ):
                findings.append(
                    f"{target.name}: Age {age}s exceeds advertised freshness "
                    f"window {stale_deadline}s"
                )

    try:
        text = response.text()
    except UnicodeDecodeError as error:
        findings.append(f"{target.name}: response is not UTF-8: {error}")
        return findings
    for fragment in target.required_fragments:
        if fragment not in text:
            findings.append(f"{target.name}: response is missing {fragment!r}")
    return findings


def googlebot_parity_findings(
    target: DiscoveryTarget,
    default: HttpResponse,
    googlebot: HttpResponse,
) -> list[str]:
    findings: list[str] = []
    if googlebot.status != default.status:
        findings.append(
            f"{target.name}: Googlebot status {googlebot.status} differs "
            f"from default status {default.status}"
        )
    if googlebot.body != default.body:
        findings.append(f"{target.name}: Googlebot body differs from default response")
    for header_name in ("cache-control", "x-robots-tag"):
        if _header(googlebot, header_name) != _header(default, header_name):
            findings.append(
                f"{target.name}: Googlebot {header_name} differs from default response"
            )
    default_content_type = _header(default, "content-type").partition(";")[0].strip()
    googlebot_content_type = (
        _header(googlebot, "content-type").partition(";")[0].strip()
    )
    if googlebot_content_type != default_content_type:
        findings.append(
            f"{target.name}: Googlebot content-type differs from default response"
        )
    if googlebot.status == 200:
        prefix = f"{target.name}: "
        for finding in discovery_response_findings(target, googlebot):
            findings.append(f"{target.name}: Googlebot {finding.removeprefix(prefix)}")
    return findings


def fetch_discovery_responses(
    targets: tuple[DiscoveryTarget, ...],
    request: Request,
) -> tuple[dict[str, HttpResponse], dict[str, HttpResponse], list[str]]:
    requests = tuple(
        (target, user_agent, label)
        for target in targets
        for user_agent, label in (
            (USER_AGENT, "default"),
            (GOOGLEBOT_USER_AGENT, "Googlebot"),
        )
    )
    results: dict[tuple[str, str], HttpResponse] = {}
    findings: list[str] = []
    with ThreadPoolExecutor(max_workers=min(16, len(requests))) as pool:
        futures = {
            pool.submit(
                request,
                "GET",
                target.url,
                headers={"Accept": "*/*", "User-Agent": user_agent},
                follow_redirects=False,
            ): (target, user_agent, label)
            for target, user_agent, label in requests
        }
        for future in as_completed(futures):
            target, user_agent, label = futures[future]
            try:
                results[(target.url, user_agent)] = future.result()
            except (OSError, TimeoutError, urllib.error.URLError) as error:
                findings.append(f"{target.name}: {label} fetch failed: {error}")

    default = {
        target.url: results[(target.url, USER_AGENT)]
        for target in targets
        if (target.url, USER_AGENT) in results
    }
    googlebot = {
        target.url: results[(target.url, GOOGLEBOT_USER_AGENT)]
        for target in targets
        if (target.url, GOOGLEBOT_USER_AGENT) in results
    }
    return default, googlebot, findings


def body_parity_findings(
    name: str,
    proxied: HttpResponse,
    canonical: HttpResponse,
) -> list[str]:
    if proxied.body == canonical.body:
        return []
    return [f"{name}: proxied body differs from canonical body"]


def agent_card_skill_findings(name: str, response: HttpResponse) -> list[str]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return [f"{name}: response is not valid JSON"]
    skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(skills, list):
        return [f"{name}: response has no skills list"]
    skill_ids = [
        skill.get("id")
        for skill in skills
        if isinstance(skill, dict) and isinstance(skill.get("id"), str)
    ]
    if len(skill_ids) == len(skills) and Counter(skill_ids) == Counter(EXPECTED_SKILLS):
        return []
    return [
        (
            f"{name}: expected one entry for each skill "
            f"{sorted(EXPECTED_SKILLS)}, got {sorted(skill_ids)}"
        )
    ]


def parse_skill_index(index: object) -> tuple[SkillEntry, ...]:
    if not isinstance(index, dict) or not isinstance(index.get("skills"), list):
        raise DiagnosticsFailure("skill index: invalid skills payload")
    raw_entries = index["skills"]
    entries: list[SkillEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise DiagnosticsFailure("skill index: invalid skill entry")
        name = raw_entry.get("name")
        url = raw_entry.get("url")
        digest = raw_entry.get("digest")
        if not all(isinstance(value, str) for value in (name, url, digest)):
            raise DiagnosticsFailure(
                "skill index: skill entry is missing name, url, or digest"
            )
        entries.append(SkillEntry(name=name, url=url, digest=digest))

    names = [entry.name for entry in entries]
    duplicate_names = sorted(
        name for name, count in Counter(names).items() if count != 1
    )
    if duplicate_names:
        raise DiagnosticsFailure(
            f"skill index: duplicate skill names: {duplicate_names}"
        )
    if set(names) != EXPECTED_SKILLS or len(entries) != len(EXPECTED_SKILLS):
        raise DiagnosticsFailure(
            "skill index: expected exactly one entry per public skill: "
            f"expected {sorted(EXPECTED_SKILLS)}, got {sorted(names)}"
        )

    urls = [entry.url for entry in entries]
    duplicate_urls = sorted(url for url, count in Counter(urls).items() if count != 1)
    if duplicate_urls:
        raise DiagnosticsFailure(f"skill index: duplicate skill URLs: {duplicate_urls}")
    for entry in entries:
        expected_url = f"/.well-known/agent-skills/{entry.name}/skill.md"
        if entry.url != expected_url:
            raise DiagnosticsFailure(
                f"skill index: {entry.name} URL must be {expected_url!r}, "
                f"got {entry.url!r}"
            )
        if DIGEST_PATTERN.fullmatch(entry.digest) is None:
            raise DiagnosticsFailure(
                f"skill index: {entry.name} has invalid SHA-256 digest"
            )
    return tuple(entries)


def skill_targets(entries: tuple[SkillEntry, ...]) -> tuple[DiscoveryTarget, ...]:
    targets: list[DiscoveryTarget] = []
    for entry in entries:
        targets.extend(
            (
                DiscoveryTarget(
                    f"canonical {entry.name} skill",
                    urllib.parse.urljoin(f"{DOCS_ORIGIN}/", entry.url),
                    ("text/markdown", "text/plain"),
                    3600,
                ),
                DiscoveryTarget(
                    f"proxied {entry.name} skill",
                    urllib.parse.urljoin(f"{AGENT_ORIGIN}/", entry.url),
                    ("text/markdown", "text/plain"),
                    3600,
                    allow_noindex=True,
                ),
            )
        )
    return tuple(targets)


def skill_digest_findings(
    entries: tuple[SkillEntry, ...],
    artifacts: dict[str, bytes],
) -> list[str]:
    findings: list[str] = []
    for entry in entries:
        artifact = artifacts.get(entry.name)
        if artifact is None:
            findings.append(f"{entry.name}: canonical skill artifact was not fetched")
            continue
        actual = f"sha256:{hashlib.sha256(artifact).hexdigest()}"
        if actual != entry.digest:
            findings.append(f"{entry.name}: artifact digest does not match skill index")
    return findings


def run_diagnostics(
    *,
    request: Request,
    validate_openapi: OpenApiValidator,
) -> int:
    targets = base_discovery_targets()
    default, googlebot, findings = fetch_discovery_responses(targets, request)

    index_url = f"{DOCS_ORIGIN}/.well-known/agent-skills/index.json"
    index_response = default.get(index_url)
    entries: tuple[SkillEntry, ...] = ()
    if index_response is not None:
        try:
            entries = parse_skill_index(index_response.json())
        except (DiagnosticsFailure, json.JSONDecodeError) as error:
            findings.append(f"canonical skill index: {error}")

    dynamic_targets = skill_targets(entries)
    if dynamic_targets:
        skill_default, skill_googlebot, skill_findings = fetch_discovery_responses(
            dynamic_targets,
            request,
        )
        default.update(skill_default)
        googlebot.update(skill_googlebot)
        findings.extend(skill_findings)
        targets += dynamic_targets

    for target in targets:
        default_response = default.get(target.url)
        googlebot_response = googlebot.get(target.url)
        if default_response is not None:
            findings.extend(discovery_response_findings(target, default_response))
        if default_response is not None and googlebot_response is not None:
            findings.extend(
                googlebot_parity_findings(
                    target,
                    default_response,
                    googlebot_response,
                )
            )

    parity_pairs = [
        (
            "agent skill index",
            f"{AGENT_ORIGIN}/.well-known/agent-skills/index.json",
            index_url,
        ),
        (
            "llms.txt",
            f"{AGENT_ORIGIN}/llms.txt",
            f"{DOCS_ORIGIN}/llms.txt",
        ),
    ]
    artifacts: dict[str, bytes] = {}
    for entry in entries:
        canonical_url = urllib.parse.urljoin(f"{DOCS_ORIGIN}/", entry.url)
        proxied_url = urllib.parse.urljoin(f"{AGENT_ORIGIN}/", entry.url)
        parity_pairs.append((f"{entry.name} skill", proxied_url, canonical_url))
        canonical = default.get(canonical_url)
        if canonical is not None:
            artifacts[entry.name] = canonical.body
    findings.extend(skill_digest_findings(entries, artifacts))

    for name, proxied_url, canonical_url in parity_pairs:
        proxied = default.get(proxied_url)
        canonical = default.get(canonical_url)
        if proxied is not None and canonical is not None:
            findings.extend(body_parity_findings(name, proxied, canonical))

    for name, url in (
        ("agent card", f"{AGENT_ORIGIN}/.well-known/agent-card.json"),
        ("API agent card", f"{API_ORIGIN}/.well-known/agent-card.json"),
    ):
        response = default.get(url)
        if response is not None:
            findings.extend(agent_card_skill_findings(name, response))

    openapi_response = default.get(f"{DOCS_ORIGIN}/openapi.json")
    if openapi_response is not None:
        try:
            document = openapi_response.json()
            if not isinstance(document, dict):
                raise DiagnosticsFailure("canonical OpenAPI must contain an object")
            validate_openapi(document)
        except (DiagnosticsFailure, RuntimeError, json.JSONDecodeError) as error:
            findings.append(f"canonical OpenAPI: {error}")

    if findings:
        raise DiagnosticsFailure("\n".join(findings))
    return len(targets)
