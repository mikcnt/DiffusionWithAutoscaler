import re
from contextlib import nullcontext
from typing import List

import torch
from lightning import LightningModule
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

NEWLINE = "Ċ"


class CodeGen(LightningModule):
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        fp16: bool = True,
        use_deepspeed: bool = True,
        enable_cuda_graph: bool = False,
    ):
        super().__init__()
        self.model = self._load_model(checkpoint_path)
        self.tokenizer = self._load_tokenizer(checkpoint_path)
        # self.device = device

        # TODO: rewrite deepspeed_injection
        # look at https://github.com/Lightning-AI/stablediffusion/blob/add_deepspeed_1/ldm/deepspeed_replace.py#L269
        # if use_deepspeed:
        #     deepspeed_injection(self.model, fp16=fp16, enable_cuda_graph=enable_cuda_graph)

        self.to(device, dtype=torch.float16 if fp16 else torch.float32)
        self.fp16 = fp16

    def predict_step(self, prompts: List[str], batch_idx: int, dataloader_idx: int = 0):
        precision_scope = torch.autocast if self.fp16 else nullcontext
        inference_context = torch.inference_mode if torch.cuda.is_available() else torch.no_grad
        with inference_context(), precision_scope(self.device.type):  # self.model.ema_scope():
            # TODO: max_length has to be set somewhere
            output = self._predict_step(prompts, max_length=100)
        return output

    def _load_model(self, checkpoint: str) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(checkpoint)

    def _load_tokenizer(self, checkpoint: str) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return tokenizer

    def _predict_step(
        self,
        batch: List[str],
        max_length: int,
        # TODO: these generate_kwargs have to be fed somewhere
        **generate_kwargs,
    ) -> List[str]:
        # Replace newlines with appropriate symbol
        # Encode batch
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        generation_max_length = max_length

        # Change the `generation_max_length` when the model repeats the prompt in the output (e.g., for CodeGen),
        # to account for the prompt length in the output of the model. In all other cases, this is ignored
        prompts_ids_shift = inputs["input_ids"].shape[1]
        generation_max_length += prompts_ids_shift

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        # Generate predictions
        tokens = self.model.generate(
            **inputs,
            max_length=generation_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        # Return decoded predictions
        # Notice that if the prompt is not repeated in output, `prompts_ids_shift` will be 0, and therefore
        # slicing will have no effect
        decoded_predictions = self.tokenizer.batch_decode(tokens[:, prompts_ids_shift:, ...])

        # Truncate outputs (e.g., for models like CodeGen)
        decoded_predictions = self.truncate_batch(decoded_predictions)

        # Replace newline symbols back to newlines
        return decoded_predictions

    def truncate(self, completion: str) -> str:
        """
        Truncate method for CodeGen predictions.
        Took from https://github.com/salesforce/CodeGen/blob/main/jaxformer/hf/sample.py#L135-L167
        """

        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        terminals = [re.compile(r, re.MULTILINE) for r in ["^#", re.escape("<|endoftext|>"), "^'''", '^"""', "\n\n\n"]]

        prints = list(re.finditer("^print", completion, re.MULTILINE))
        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        defs = list(re.finditer("^def", completion, re.MULTILINE))
        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]
        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion

    def truncate_batch(self, completion_batch: List[str]) -> List[str]:
        return [self.truncate(completion) for completion in completion_batch]
