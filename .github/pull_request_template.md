## What changed and why
<!-- One or two sentences. Link the issue or benchmark that motivated it. -->

## Area
- [ ] firmware (`firmware/`)
- [ ] server (`server/`)
- [ ] training (`training/`)
- [ ] benchmarks / tools
- [ ] docs

## How it was checked
<!-- Board run, host test, benchmark script and result file, or "docs only". -->

## Checklist
- [ ] No Wi-Fi name/password, token, `sdkconfig`, `.env`, recording (`*.wav`) or raw serial log is included
- [ ] If firmware changed: builds with ESP-IDF 5.5, strict RAM (`KWS_RAM_LIMIT_KB`) and CPU numbers re-measured or noted as unchanged
- [ ] READMEs / `benchmarks/REPORT.md` updated when behaviour or numbers changed
