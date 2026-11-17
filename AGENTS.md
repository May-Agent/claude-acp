# AGENTS.md

## Workspace layout

- `src/` — `claude-acp` crate implementing Agent Client Protocol integration over `aither-acp` abstractions.
- `sdk/` — `claude-acp-sdk` crate implementing the stream-json control protocol client over `claude-codes` (`types`) wire models.

## Build, lint, and test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --workspace --all-features --no-tests=pass
cargo test --doc --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

## Conventions

- Library code is runtime-agnostic using `futures-lite` and `async-*` primitives; never add `tokio` to library dependencies.
- Public trait async methods return `impl Future` (never `Pin<Box<dyn Future>>`).
- Errors are `thiserror` enums; `anyhow` is banned in library code.
- Diagnostics go through `tracing`, never `println!`.
- Dependencies declare `default-features = false` with only the features used.
