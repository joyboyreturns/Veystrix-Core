# Starting Veystrix manually (safe commands)

Never launch the engine unsupervised; an unsupervised direct launch once exhausted memory and froze the Mac.
Plug in AC power, switch Low Power Mode off, close heavy apps.

```sh
cd ~/Documents/Veystrix
./serve.sh --status                                   # read-only: what the supervisor did last
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2         # one supervised run; Ctrl-C stops it
VEYSTRIX_API_KEY=local ./serve.sh --overnight         # supervised, restarts after a guard stop (bounded)
curl -s http://127.0.0.1:8080/health ; curl -s http://127.0.0.1:8080/v1/models
```

The server listens on `127.0.0.1` only. Checkpoint default: the owner's configured `qwen4_exp` folder; override with
`--checkpoint DIR`. These commands load the real model: **do not run them while the owner is away.**

The macOS app (`dist/Veystrix.app` from `packaging/macos/build_app.sh`) never starts the engine by itself: use
Start in the Dashboard after choosing a model folder in Settings.
