# Capture research API integration

Mounted /api/type2branch-capture in the isolated localhost research app. Explicit true lengths accompany DEV feature retrieval and scoring; the router validates release integrity, all79accounts, unique eligible probe identities, window0, length-specific coverage and strict request keys. Train/test retrieval is rejected before release loading. Release pin remains None.

The client reuses bounded serialized IPC and process cleanup, verifies release handshakes and each response ID, and recomputes per-length threshold decisions. Generated subprocess tests cover mixed lengths, process reuse, response mismatch, wrong decisions, exit, timeout and pre-start padding rejection. Shutdown now closes both research workers.

51 combined API/client/regression tests passed in1.95seconds under approved in-process execution; two existing deprecation warnings remain. No server was launched. The full replay script retrieves every eligible DEV probe, performs actual-worker POST inference, checks all scores within1e-5 and zero decision flips, recomputes coverage/metrics, and checks split/Host/Origin/client restrictions. Its premature invocation rejected the absent release pin. Actual trained-model replay remains pending.
