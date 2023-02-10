# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
import lightning as L
import torch

from llm_with_autoscaler import AutoScaler, BatchModelOutput, BatchPrompt, ModelOutput, Prompt
from model import CodeGen

PROXY_URL = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"


class FlashAttentionBuildConfig(L.BuildConfig):
    def build_commands(self):
        return ["pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'"]


class LanguageModelServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchPrompt,
            output_type=BatchModelOutput,
            # cloud_build_config=FlashAttentionBuildConfig(),
            *args,
            **kwargs,
        )

    def setup(self):
        # TODO: download model and tokenizer here, don't use transformers api
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CodeGen(
            checkpoint_path="Salesforce/codegen-350M-multi",
            device=torch.device(device),
            fp16=False,
            use_deepspeed=True,  # Supported on Ampere and RTX, skipped otherwise.
            enable_cuda_graph=False,
        )

    def predict(self, requests):
        texts = [request.text for request in requests.inputs]
        outputs = self._model.predict_step(prompts=texts, batch_idx=0)
        return BatchModelOutput(outputs=[{"text": result} for result in outputs])


component = AutoScaler(
    LanguageModelServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=3,
    timeout_batching=0.3,
    input_type=Prompt,
    output_type=ModelOutput,
    # cold_start_proxy=CustomColdStartProxy(proxy_url=PROXY_URL),
)

app = L.LightningApp(component)
