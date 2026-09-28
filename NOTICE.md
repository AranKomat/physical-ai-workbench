# Attribution / provenance

The newly written workbench code is original glue/reference implementation for this project. No pretrained weights or full third-party repository copies are included.

Design and source-interface references include:

- AMD-AGI/Primus for the intended ROCm training substrate.
- InternRobotics/InternW0-Delta for the inspected ActionDiT signature, checkpoint guide, architecture config and native flow convention.
- InternRobotics/InternVLA-A-series as a Qwen-native VLA reference.
- sii-research/tau-0-vla for proposal HTTP fields and world-model inference/fine-tuning CLI contracts.
- Physical Intelligence's knowledge-insulation work for the stop-motor-gradient plus separate backbone-objective design.
- Qwen-RobotManip for the physical-alignment motivation.
- Hugging Face Qwen/Transformers and Diffusers for optional local native-model adapters.

Exact observed repository revisions are in upstreams.lock.json. References and verification scope are in docs/HANDOFF_V5.md. The workbench's tiny model, scalar auxiliary codec, 80D slot layout and layer mixer are NOT branded reproductions of these projects.

Third-party source, weights and datasets fetched later retain their own notices and terms. The MIT license here applies only to the newly written workbench code, not to those dependencies or generated derivative weights.
