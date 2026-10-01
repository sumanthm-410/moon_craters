# Kaggle access — verified setup instructions

**Never paste a Kaggle API key, token or `kaggle.json` content into the chat.**
Nothing in this document asks you to. No credential is ever written into this
repository, a notebook, a dataset archive or a log.

## What the installed client actually supports (VERIFIED)
Determined on 2026-10-01 by reading the source of the installed client, not
from memory. Client: **kaggle 2.2.4** (from PyPI).

`KaggleApi.authenticate()` tries, in this order:
1. an active **access token**,
2. a **legacy API key** (`kaggle.json`),
3. **OAuth** credentials,
4. anonymous fallback, where the command allows it.

The client's own help text recommends OAuth:

```
Recommended: log in with OAuth via a web-based authorization flow.
No token to manage; credentials are cached locally for you.
    kaggle auth login
```

> Note: the project brief assumed the legacy `kaggle.json` route. That route
> still works in 2.2.4, but it is no longer the client's recommended method.
> Both are documented below; pick one.

Relevant configuration names found in the client: `KAGGLE_CONFIG_DIR`,
`KAGGLE_API_TOKEN`, and a `kaggle.json` config file.

## Option 1 — OAuth (client-recommended)
Run locally, on a machine with a browser:
```bash
kaggle auth login
```
This opens a browser authorization flow and caches credentials locally.
There is no token for you to handle, store or paste.

## Option 2 — Access token
Generate a token at `https://www.kaggle.com/settings/api`
("Generate New Token" under "API"), then EITHER:
```bash
export KAGGLE_API_TOKEN=<token>          # set in your shell, not in a file in this repo
```
or save it to `~/.kaggle/access_token` and restrict it:
```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
# write the token into ~/.kaggle/access_token with your editor
chmod 600 ~/.kaggle/access_token
```

## Option 3 — Legacy `kaggle.json`
```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
The file stays at `~/.kaggle/kaggle.json`. It is never copied into the
repository, printed, logged or uploaded. `.gitignore` excludes it defensively.

## Verifying authentication harmlessly
Once credentials are in place, verify with a read-only call that changes
nothing:
```bash
kaggle config view          # shows configuration, not secrets
kaggle datasets list --mine --page-size 1
```
A successful listing proves authentication. Do not verify by uploading.

## Status in THIS environment
**BLOCKED, not configured.** `www.kaggle.com` is denied by the environment's
network policy (CONNECT 403, verified 2026-10-01). No authentication method
can be exercised or verified from this container until that is changed, and
OAuth additionally needs an interactive browser. Authentication has therefore
NOT been tested here and no claim is made that it works.

## Visibility policy for this project
Datasets and notebooks are created **private** by default. After creation,
actual visibility is re-read from the API and recorded — the requested setting
is not taken as proof of the result.

## Secret handling in notebooks
Remote notebooks use Kaggle's own secrets mechanism (Add-ons -> Secrets),
read at runtime. Credentials are never packaged into notebook source or into
a dataset archive. `download.py` redacts token-like query parameters from all
error messages and logs.
