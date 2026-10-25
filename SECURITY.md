# Security policy

KhmerOCR processes images of identity documents, so privacy bugs count as
security bugs.

## Reporting a vulnerability

Report privately through GitHub:
[**Report a vulnerability**](https://github.com/Alujack/cambotix.ocr/security/advisories/new).
Please do not open a public issue.

Include the affected package and version, what an attacker can do, and steps
to reproduce. **Never attach real ID card images or anyone's personal data;**
use the synthetic SPECIMEN cards from `khmerocr.testing` instead.

This is a volunteer-maintained project. Reports are acknowledged as soon as
possible, and fixes for confirmed issues are released before details are
made public.

## In scope

- Document data leaving the machine: network access by the engine (it must
  work fully offline), logs, temporary files, or error messages containing
  document content.
- Bypassing the REST API's key check (`KHMEROCR_API_KEY`) or upload limits.
- Unsafe handling of untrusted images or model bundles (path traversal,
  decompression bombs, checksum bypass).
- The TypeScript and Dart clients leaking keys or image data.

## Out of scope

- OCR mistakes. Results are designed to be reviewed by a person; report
  accuracy problems as normal issues.
- Deployments that expose the API to a network without `KHMEROCR_API_KEY`
  and HTTPS, against the documentation.

## Supported versions

Only the latest release of each package receives fixes while the project is
in 0.x.
