# Project rules

Follow the execution plan and style guide in the two root Word documents. Complete phases in order. NEVER auto-advance to the next phase: run the current phase's verification gate, stop, and wait for user confirmation.

Python 3.11 or newer is acceptable. For Phase 0, SUPABASE_READONLY_URL may remain blank until the read-only role is created in Phase 1, and missing npm is a warning until Phase 8.

## Safety

1. Run generated Python only in the Docker or E2B sandbox, never with host exec, eval, or subprocess.
2. Use a read-only PostgreSQL role and sqlglot SELECT-only validation for SQL tools.
3. Require a LangGraph interrupt and explicit confirmation before every refund, ticket, email, or other write action.
4. Keep the Render web container below 512 MB; do not load local PyTorch, embedding, or Kokoro weights there.
5. Never commit .env, API keys, or generated chart PNGs; pin dependency versions.
6. Cap self-repair at three retries and sandbox execution at 30 seconds.

## Style

Use small, flat modules, direct functions, plain names, simple control flow, and specific error handling. Keep comments and docstrings short and useful. Avoid unused code, unnecessary abstractions, buzzwords, emojis, and placeholder stubs. Create only planned files. Use a plain light-mode UI with restrained colors and minimal dependencies. Keep documentation and commit messages short and factual. Run the phase gate and report its actual result.
