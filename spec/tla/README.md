# Order-lifecycle model checks

Run `bash scripts/run_tlc.sh` with Java 21, Bash, curl and GNU coreutils.
The script checks safety and liveness separately without changing their bounds,
invariants or properties. No Python application code is required.

## Pinned TLC artifact

We use stable **TLA+ tools v1.7.4 / TLC 2.19** (revision `5a47802`).
The upstream `v1.8.0` tag is a rolling prerelease: its build workflow replaces
`tla2tools.jar` and force-moves the tag. A version-looking download URL therefore
cannot be paired reliably with a fixed checksum.

- Official release: <https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4>
- Fixed asset metadata/download endpoint:
  <https://api.github.com/repos/tlaplus/tlaplus/releases/assets/184694200>
- Size: `2274532` bytes
- SHA-256: `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`
- Upstream release's published SHA-1:
  `bee4a54f3ee3d4afc347c3240ec2d9e93b075104`

On 2026-10-02 the artifact downloaded over HTTPS from the official asset API
matched the independently published release checksum; its SHA-256 and size were
then calculated locally. The old release API has no SHA-256 digest field value.
SHA-1 is provenance evidence only; runtime verification requires the pinned
SHA-256 **and** size, for both downloads and `TLA_TOOLS_JAR` overrides.

GitHub documents downloading asset bytes with `Accept: application/octet-stream`:
<https://docs.github.com/en/rest/releases/assets#get-a-release-asset>.
Its asset-update API changes metadata, not bytes. An asset-ID URL does not follow
a replacement upload. If the asset is deleted, download fails closed; there is
no fallback to a moving tag or automatic checksum update.

Compatibility was checked with both existing configs on Java 21: safety generated
3,523 states (345 distinct); liveness generated 428 (76 distinct). Both completed
without errors. v1.7.4 includes upstream's multi-worker liveness soundness fix
(<https://github.com/tlaplus/tlaplus/issues/971>). The model uses standard integer,
set, record and temporal operators; it does not require prerelease extensions.

Downloads use a unique temporary file beside the cache and are verified before
rename. Failed/partial downloads are removed and never cached or executed.
`TLA_TOOLS_JAR` must name a file, not a directory; a directory target fails
without retaining the temporary download. GNU `mv -T` prevents directory targets
from absorbing the downloaded file.
An invalid existing cache is reported with actual/expected digests and sizes,
not silently replaced. Remove that file deliberately after investigating it.
The default versioned filename avoids reusing older `.tla/tla2tools.jar` caches.

Anonymous `api.github.com` requests share a 60/hour limit per IP, and shared
CI runners exhaust it (`curl: (22) ... 403`). When `GITHUB_TOKEN` is set, the
download authenticates from a temporary 0600 header file, so the token never
appears in argv; curl drops the header on the cross-host redirect to the asset
store. CI passes the job token and caches `.tla/` under a key that embeds the
pinned SHA-256; cached bytes still pass the same digest and size check.

For upgrades, establish official artifact provenance and a fixed asset ID,
verify bytes independently before running them, update SHA-256/size/version
alongside this record, and rerun both model configs and the offline downloader
regression tests (`python scripts/test_run_tlc.py`). Never repair a checksum
failure by trusting the failed download or weakening verification.
