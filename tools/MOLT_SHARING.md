# Sharing the dashboard and GPU steering

The existing layer-22 GPU server remains at `100.76.58.1:8765`. A separate Caddy
password gate listens on `127.0.0.1:8786` and proxies both the static explorer and
all steering API routes. It uses the same running RTX PRO 6000 model and existing
single-job inference lock. No additional model is loaded.

The proxy is installed and enabled as the user service
`molt-dashboard-share.service`. Caddy 2.11.4 is installed under
`~/.local/lib/molt-dashboard-sharing/2.11.4/`; its archive was verified against the
official release's SHA-512 checksum list.

## Enable the public HTTPS link

The Tailscale daemon requires administrator privileges to change its serving
configuration on this machine. Run locally:

```bash
sudo tailscale funnel --bg --https=443 http://127.0.0.1:8786
```

The expected public address is `https://amsterdam.taild80f17.ts.net/`. If Tailscale
requests a one-time web setup, complete the URL it prints. Check publication with
`tailscale funnel status`.

The browser asks for the username `friends` and the generated password stored in
`~/.config/molt-dashboard-sharing/credentials.json` (mode 0600). That file is
outside the dashboard's served directory. The private Caddyfile stores only the
password's bcrypt hash. Its source template is `tools/molt_dashboard_share.Caddyfile`.

All pages and API routes require the password. The proxy preserves Host and
Origin for the API's origin checks, removes the Authorization header before
forwarding upstream, and blocks extraction caches and hidden worker files.

## Verification

`tests/browser/dashboard_sharing.cjs` reads the private credentials file and
checks authentication, the 10M-token explorer, all four scores, embedded activation
examples, the steering UI, API access, and rejection of cross-origin writes.
By default it uses the HTTPS URL in that file. `MOLT_SHARE_URL` can select another
address, such as `http://127.0.0.1:8786/` for local proxy checks.

Set `MOLT_STEERING_SMOKE=1` to additionally run one real four-token baseline/steered
comparison and verify a nonzero gate intervention and final-logit change. Otherwise
the test performs no GPU inference. It requires the same Node/Playwright setup as
the other browser tests, with `CHROMIUM_PATH` selecting the browser executable.

Local browser and GPU smoke checks passed during setup. Funnel was enabled on
September 19, 2026; the browser checks and real GPU steering comparison also passed
at the HTTPS address. Requests forced through a public Funnel relay verified that
the site is reachable outside Tailscale, rejects anonymous access, and accepts
authenticated page, health, and tokenization requests with valid TLS.

## Operation

```bash
systemctl --user status molt-dashboard-share.service
journalctl --user -u molt-dashboard-share.service -n 30
tailscale funnel status
```

To close public access while retaining the local dashboard:

```bash
sudo tailscale funnel --https=443 off
```

Funnel in background mode and the password gate persist across reboots. The
existing GPU server is a transient service, so after reboot it must be started
again before steering and proxied pages are available.
