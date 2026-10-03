# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/mrcabbage972/corpus-assay/security/advisories/new)
rather than opening a public issue. Include a description, the affected version, and
steps to reproduce. You should receive a response within a week.

## Supported versions

Only the latest released version receives security fixes.

## Scope notes

corpus-assay loads datasets through the Hugging Face `datasets` library. Only scan or
index data you trust, and do not enable `--trust-remote-code` for datasets whose
loading scripts you have not reviewed.
