#!/usr/bin/env python3
"""Send CI status notification to Feishu via Bot API."""

import json
import os
import sys
import urllib.request


def main():
    app_id = os.environ["FEISHU_APP_ID"]
    app_secret = os.environ["FEISHU_APP_SECRET"]
    chat_id = os.environ["FEISHU_CHAT_ID"]
    ci_status = os.environ["CI_STATUS"].strip()
    workflow = os.environ["WORKFLOW_NAME"]
    platform = os.environ.get("PLATFORM", "")
    repo = os.environ["REPO"]
    ref = os.environ["REF"]
    sha = os.environ["SHA"]
    actor = os.environ["ACTOR"]
    run_id = os.environ["RUN_ID"]
    server_url = os.environ["SERVER_URL"]
    event_name = os.environ["EVENT_NAME"]

    short_sha = sha[:7]
    run_url = f"{server_url}/{repo}/actions/runs/{run_id}"

    # --- 1. Obtain tenant_access_token ---
    token_req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(token_req) as resp:
        token_data = json.loads(resp.read())

    tenant_token = token_data.get("tenant_access_token", "")
    if not tenant_token:
        print(
            f"::error::Failed to obtain tenant_access_token: {token_data}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 2. Build message card ---
    status_map = {
        "success": ("\u2705", "Success", "green"),
        "cancelled": ("\u26a0\ufe0f", "Cancelled", "orange"),
    }
    emoji, text, color = status_map.get(ci_status, ("\u274c", "Failed", "red"))

    title = f"CI {text} \u2014 {repo}"
    if platform:
        title = f"CI {text} [{platform}] \u2014 {repo}"

    card_content = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Workflow:** {workflow}",
                        },
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Status:** {emoji} {text}",
                        },
                    },
                    {
                        "is_short": True,
                        "text": {"tag": "lark_md", "content": f"**Branch:** {ref}"},
                    },
                    {
                        "is_short": True,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Trigger:** {event_name} by {actor}",
                        },
                    },
                    {
                        "is_short": False,
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Commit:** {short_sha}",
                        },
                    },
                ],
            },
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "View Run"},
                        "url": run_url,
                        "type": "primary",
                    }
                ],
            },
        ],
    }

    body = json.dumps(
        {
            "receive_id": chat_id,
            "msg_type": "interactive",
            "content": json.dumps(card_content),
        }
    ).encode()

    # --- 3. Send message ---
    msg_req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
        data=body,
        headers={
            "Authorization": f"Bearer {tenant_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(msg_req) as resp:
        result = json.loads(resp.read())

    if result.get("code", -1) != 0:
        print(f"::error::Feishu API returned error: {result}", file=sys.stderr)
        sys.exit(1)

    print("Feishu notification sent successfully.")


if __name__ == "__main__":
    main()
