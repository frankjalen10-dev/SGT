# SGT: Securing Open-Source LLMs Against Malicious Fine-tuning via Safety Guidance Trigger
This repository contains the official implementation of the paper "SGT: Securing Open-Source LLMs Against Malicious Fine-tuning via Safety Guidance Trigger".

SGT is a robust defense framework designed to safeguard open-weight LLMs against Malicious Fine-Tuning (MFT). By leveraging a learnable Safety Guidance Trigger, SGT anchors model representations into a resilient safety region. This structural alignment ensures that the model preserves its safety mechanisms and consistently rejects malicious instructions, even after malicious fine-tuning.

# Key Features
Defense against Malicious Fine-tuning (MFT): Provides a robust defense framework that preserves the safety alignment of open-weight LLMs even under adversarial attacks.
Structural Alignment via SGT: Implements a Safety-Guided Trigger (SGT) mechanism that anchors model representations to a designated safety manifold, preventing drift.
Robust Alignment Persistence: Significantly raises the barrier for attackers by ensuring consistent mapping of malicious prompts to safe regions, outperforming traditional parameter-level defenses.
License
SGT and its family are released under the CC BY-NC 4.0 License.

# Repository Structure
.
├── 3.1_stage1_learning_safety_guidance_trigger.py    # [Core] Script for optimizing the Safety Guidance Trigger (SGT)
├── 3.2_stage2_trigger_guided_representation_alignment # [Core] Script for align the Safety Guidance Trigger (SGT)
├── 4.2_mft.py                  # Script for simulating malicious fine-tuning 
├── 4.2_eval_mft_model.py       # Evaluation script: Calculates Harmfulness Score using GPT-5.1 judge
├── 4.3_eval_benign.py          # Evaluation script: Evaluate the model's general capabilities(mmlu, OpenBookQA) using lm-evaluation-harness
├── utils/
│   ├── common.py               # Common utilities (Seed, Padding)
│   └── soft_trigger.py         # Implementation of trigger insertion
└── README.md