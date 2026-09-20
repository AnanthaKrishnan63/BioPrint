# Research API integration audit

Mounted /api/type2branch on the isolated research app. Release pin remainsNone. Worker client serializes requests, enforces response IDs/shapes/finite scores/float64decisions, bounds pipe I/O and deadlines, and terminates on errors or app lifespan shutdown. No TensorFlow import in parent.

33combined generated/in-process tests passed under approved execution. Initial old FastAPI shutdown API was unavailable; replaced with lifespan cleanup. Sandbox AnyIO threadpool tests stalled, while unchanged suite passed outside sandbox. ActualDEVreplay script retrieves every probe through GET and submits it through POST, checking full score/decision/metric parity and local-only/test-seal gates. This replay remains unexecuted until a completed release is pinned.
