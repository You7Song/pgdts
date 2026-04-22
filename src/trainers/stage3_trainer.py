"""
PGDTS Stage 3 Trainer — Decision Generation
============================================
Standard autoregressive fine-tuning (SFT) for generating concise,
action-oriented avoidance instructions.

Loss: L_Stage3 = L_AR^decision

No custom auxiliary loss is required; the hierarchical perception-decision
association built in Stages 1 & 2 is leveraged implicitly through the
structured perception inputs in the prompt.
"""

from .pgdts_trainer_base import PGDTSDataset, PGDTSTrainerBase


class Stage3Trainer(PGDTSTrainerBase):
    """
    Stage 3 trainer — pure supervised fine-tuning with standard AR loss.
    """

    def _build_dataset(self, jsonl_path: str):
        return PGDTSDataset(
            jsonl_path=jsonl_path,
            processor=self.processor,
            system_prompt=self.system_prompt,
            max_length=self.pgdts_args.max_length,
        )

    # compute_loss is inherited from PGDTSTrainerBase (standard AR loss)
