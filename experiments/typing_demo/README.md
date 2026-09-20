# Trained typing demonstration

From the main-branch repository root, with the Conda environment activated:

```bash
python -m pip install -r requirements-dev.txt
bash experiments/typing_demo/run.sh
```

On first run, the script downloads CMU and reconstructs the checksum-verified cohort. It uses the activated environment, creates temporary synthetic accounts, enrolls them, trains the compatible radial basis function support vector machine, and calls the real login API. It checks scores against saved evidence and verifies session issuance, then opens a local HTML results page. Use `--no-open` on a headless machine. It does not start a listener or touch the live app database.

The four primary examples are true accept, true reject, false reject and false accept **of the typing classifier** at its model threshold. The page separately shows the full app decision. In the ordinary impostor false-accept example, the stricter direct-login threshold requests additional verification rather than issuing a session. A fifth owner-like typing attack demonstrates an actual full-login false accept under that explicit attack assumption.

Examples are transparently selected from existing development outcomes, not newly generated to hit a desired score. They demonstrate failure modes, not accuracy. No model settings or thresholds are changed. The password `.tie5Roanl` is public dataset text; use no personal credentials. Results are written under `experiments/runtime/typing-demo/`.
