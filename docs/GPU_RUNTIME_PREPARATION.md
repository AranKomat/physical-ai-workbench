# First NVIDIA runtime

Do not provision hardware until the user supplies it. Do not modify another
workload's environment or replace a working ROCm torch installation.

On the future Linux NVIDIA host, create an isolated Python 3.12 environment.
Install torch 2.7.1, torchvision 0.22.1 and, if using it, torchcodec 0.5.0 from
the CUDA 12.8 wheel index as one compatible set. Install the pinned model/data
dependencies from `configs/tau-cpu-requirements.txt` while constraining those
torch wheels; the Mac environment itself is not portable.

LeRobot 0.4.1 must be installed separately with `--no-deps` under the recorded
Hub/PyArrow overrides. Preserve these deviations in the run manifest. The
current CPU data tests use PyAV; qualify the selected GPU-host decoder rather
than assuming TorchCodec parity. W0 source imports also require OmegaConf 2.3.0.
Do not install A1.5's LeRobot fork into the Tau environment: give it its own
environment and source path.

1. Verify GPU, driver, CUDA runtime, free/total VRAM, source pins and hashes.
2. Run native imports and CPU fixture/mask tests on that host.
3. Import the Tau ModelBuilder and W0 ActionDiT; start with eager attention.
   Flash/FLA kernels are a later optimization, not a reason to silently change
   the numerical baseline. If upstream requires one, record and qualify it.
4. Run `prepare_gpu_bundle.py` locally; transfer its listed files and clone
   pinned source revisions. Do not upload credentials, caches or entire home
   directories. Large checkpoint files are not stored in the public Git repo.
5. Dry-run `run_nvidia_window.py` against the generated plan. Its readiness flag
   remains false until host preflight passes; then record that evidence and
   explicitly promote the plan before `--execute`.
6. Run bounded native Tau observation probes, then the standalone W0 numeric
   probe. Review outputs and memory before opening any longer experiment.

The W0 numeric probe captures velocity, selected gradients, an SGD update and a
four-step sampler trace. It deliberately uses fixed synthetic inputs and newly
initialized context/state interfaces. It is not a reproduction of the intact
W0 MoT policy. Its source action weights must pass a strict checkpoint audit.

A1.5 native execution is separately gated on its full runtime and prefix-cache
qualification. A single generic final-layer bridge cannot replace its native
per-layer Qwen coupling without becoming a new transfer experiment. Do not
advertise a prepared A1.5 checkpoint inventory as an implemented trained bridge.
