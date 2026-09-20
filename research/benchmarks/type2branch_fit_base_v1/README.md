# KeyRecs fitting base channels

All47frozen fitting identities contribute their first15full nonoverlapping
100-row windows (705sequences,70,500events). Original session1 only. Later
windows are not encoder inputs. Selection/calibration/DEV/test values were not
decoded. Raw source and code/role hashes are pinned; no identity substituted.

This is an explicit KeyRecs adaptation. It preserves published row order,
checks inferred-next-hold and known-key adjacency, maps malformed key prefixes
to unknown0 while marking unchecked adjacencies, and accepts only whole-ms
values within1e-6ms numerical tolerance. An observed final-row pattern retains
the current hold/key event; this is not a confirmed collector specification.

`base` has VK/255,HTseconds,previous-DDseconds. It follows author's code:
negative timing clamp0, upper clip30s, first FT0 for each window. The paper's
10s clipping discrepancy remains explicit. Signed raw incoming timings,
including original intervals at window boundaries, are stored separately.
True length100 is explicit; zeros must not be used to infer padding/length.

All1,302full windows from fitting sessions matched the extracted author's
prepare_sample function exactly using relative timestamp fixtures derived
from the same signed intervals. This proves the transformation arithmetic,
not original timestamp/collector semantics or exact SOTA reproduction.

Two synthetic residual channels remain absent. Do not feed these3-channel
arrays to the5-channel model or claim full training/recognition validation.
Four generated adapter tests pass; independent review informed adjacency,
terminal-current-key and original-boundary-interval checks.
