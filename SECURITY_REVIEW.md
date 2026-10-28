# Security and quality review of GreenPulse

Date: 28 October 2026. Scope: the whole `traffic-signal` project (simulator, tuning code, tests, web application).
GreenPulse is a single-user demonstration that runs on one computer. It has no login, no database, no uploads and no network calls of its own,
so the realistic risks are an outdated web framework, an app that listens on the network without need, and crashes on odd input.

## How it was checked

| Check | Tool or method | Result |
|---|---|---|
| Bugs and unsafe constructs in the code | `ruff` with the bug (F, B), error (E9, PLE) and security (S, the bandit rules) rule sets over `src`, `app`, `experiments`, `tests` | no security finding; only unused variables and style notes, cleaned up |
| Known vulnerabilities in the libraries | `pip-audit` against the PyPI and OSV advisory databases, run twice: on the old setup, then again on the upgraded one | 3 advisories in Streamlit 1.33.0 and more in its web-server packages (in the second run also in the packages that the newer Streamlit adds); none left in the 51 packages the app depends on after the upgrade (see below) |
| The web application | code review for HTML and script injection, file access, input limits, network exposure and error handling | 5 findings, all fixed |
| Errors at run time | 8 tests with Streamlit's own test harness (`tests/test_app.py`: every demand pattern, every controller, every slider at its limits, zero demand, the Pareto lab) on Streamlit 1.33.0 and 1.66.1, plus manual checks in a browser at desktop and phone width | 1 crash and 1 logic error found and fixed; 0 failures now |
| The simulator | runs with zero demand, one car per hour, the maximum demand, and maximum sensor noise with rain, for all 8 controllers | no NaN, no infinity, no exception |

## Findings

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | `requirements.txt` pinned Streamlit 1.33.0. It has three advisories: CVE-2024-42474 (path traversal on Windows through the static-file feature, fixed in 1.37.0), CVE-2026-33682 (unauthenticated request forgery that can expose Windows credentials, fixed in 1.54.0) and CVE-2026-10804 (weak hash in the data cache, low, fixed in 1.53.1). | medium | **Fixed**: `streamlit>=1.54.0`. The app was tested on 1.66.1 and still runs on 1.33.0. |
| 2 | The server packages Streamlit uses were old: `tornado` 6.5.2 (23 advisories), `jinja2` 3.1.4 (3), `gitpython` 3.1.46 (46), `urllib3` 2.7.0 (3), `pillow` 12.3.0 (39), `certifi` 2026.6.17 (1). The second audit, on the environment with Streamlit 1.66.1, found three more: `starlette` 1.0.0, the web server of the newer Streamlit (10 entries, 5 different issues: Host-header and path injection into `request.url`, an outbound-connection trick through `StaticFiles` on Windows, handler lookup by HTTP method, and a form-size limit that was ignored; fixed in 1.3.1), `anyio` 4.13.0 (2, fixed in 4.14.2) and `fonttools` 4.60.0 (1, a file-write flaw in a script that GreenPulse never runs; fixed in 4.60.2). | medium | **Fixed**: minimum versions raised in `requirements.txt` (all of the above plus `uvicorn`). After upgrading to starlette 1.10.0, anyio 4.15.1 and fonttools 4.68.0, `pip-audit` reports no advisory in any of the 51 packages the app depends on, and all 20 tests pass. |
| 3 | The app listened on every network interface (the start-up message showed a network URL and the public address), so anyone who could reach the computer could open it. | medium | **Fixed** in the course-submission project, whose `.streamlit/config.toml` sets `address = "127.0.0.1"`, keeps the cross-origin and XSRF protections on, turns the static-file route off and stops usage statistics (checked on both Streamlit versions: the start-up message shows only `http://localhost:<port>` and, on 1.66.1, `Uvicorn server started on 127.0.0.1`). The file in this repository leaves `address` out, because the hosting platform has to reach the app over the network; for a local copy start it with `streamlit run app/app.py --server.address 127.0.0.1`. |
| 4 | With zero demand the Compare tab divided by the delay of the reference controller, which is zero, and the whole page crashed with `ZeroDivisionError`. | low (self-inflicted denial of service) | **Fixed**: the percentage change returns "no value" when the reference is zero. Covered by `test_custom_demand_extremes`. |
| 5 | In the Pareto lab the button "Use this controller" saved the choice after the controller list had been built, so the new entry did not appear until some other widget was touched. | low (logic error) | **Fixed**: the choice is now saved in a button callback, which runs before the page is built. Covered by `test_pareto_lab_member_can_be_used_as_controller`. |
| 6 | Text was put into HTML and into the animation page (built from a template and sent to the browser) without escaping. Today everything inserted is a number or a fixed string, so nothing could be exploited, but one later change could open a hole. | low | **Hardened**: dynamic text is HTML-escaped, `<` is escaped inside the embedded JSON, the animation script only uses `textContent` (never `innerHTML`), and everything embedded is numbers or digit strings. |
| 7 | The sixth demand pattern (random mix) was missing from the app although the report uses it. | low (missing feature) | **Fixed**: added to the Demand list. |

## What was found to be fine

- No `eval`, `exec`, `pickle`, shell call, or download anywhere in the app. Results are read with `json`; `numpy.load` is used with its safe default (no pickles).
- Only fixed files inside the project are read; no path comes from the user.
- Every input is a bounded widget (demand 0 to 700 cars/h and a multiplier up to 1.5, sensor noise up to 5 cars, 40% missed detections, seeds 1 to 9999, at most 12 random streams, at most 15 animated minutes). One page update takes well under a second.
- The environment variable `GREENPULSE_MIN_GREEN_LO` that the tuning scripts use is converted with `float()`; a wrong value stops the script at start-up and nothing else.

## Residual risks and advice

- Streamlit embeds the animation in a frame that allows scripts and same-origin access (the browser console says so). That is acceptable because the page is our own file plus numbers. **Never pass text or HTML that a user typed into `embed()`.**
- There is no login. If you ever put the app on a public server, add authentication and HTTPS in front of it (a reverse proxy or the hosting platform), and do not change `address` to `0.0.0.0` on a shared network.
- New advisories appear all the time. Re-run the checks before a demo or a deployment:

```bash
python -m pip install -r requirements.txt
python -m pip_audit
python -m ruff check src app experiments tests --select E9,F,B,S,PLE
python tests/test_app.py
```

- `pip-audit` audits every package in the Python installation, not only this project. The development PC has outdated packages that this project does not use (for example `pip`, `setuptools`, `cryptography`, `aiohttp`, `jupyterlab`); updating them is worth doing but is outside the project. To audit only what GreenPulse needs, create a clean environment first: `python -m venv venv`, activate it, `pip install -r requirements.txt`, then `pip-audit`.
