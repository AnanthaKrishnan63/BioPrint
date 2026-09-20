# Calibration runner gate audit

Nineteen generated tests cover completed-training gates, checkpoint minimum-loss selection and earliest ties, source-compatible scoring, float64threshold handling, and perfect/tied metrics. Tests run with both general NumPy2.5 and TensorFlow overlayNumPy1.26. The executable was invoked against the currently incomplete run: it refused before calibration artifact creation and loaded the pinned NumPy1.26.4.

Independent review identified the initial import-order bug; fixed before actual calibration. Role manifest hashes now checked against feature-preparation plan; runtime restore must match recorded optimizer iteration count. Calibration metrics are labeled innerTRAIN; no model choice can use them.

Actual checkpoint restore, inference, NPZ serialization, and recognition results are pending completed training. Source/runtime validation here does not prove that later inference passes.
