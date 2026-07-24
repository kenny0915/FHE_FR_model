## Execution environment

- Full training runs on a separate GPU server with 4 NVIDIA V100-SXM2-32GB GPUs.
- This repository checkout is only for code modifications and small, lightweight tests.
- Do not launch full training jobs in this environment; prepare and validate the code here, then run training on the GPU server.

# Repository working instructions

- When the user asks to fix, modify, or generate repository content, finish the requested work and run appropriate tests.
- After the tests confirm the requested change works, commit all files belonging to that request and push the commit to the current branch's configured upstream, unless the user explicitly asks not to commit or push.
- Do not include unrelated working-tree changes in the commit.
