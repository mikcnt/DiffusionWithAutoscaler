# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
# !pip install 'git+https://github.com/Lightning-AI/lightning.git'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
import asyncio
import base64
import io
import os
import traceback
import uuid

import lightning as L
import torch

from llm_with_autoscaler import AutoScaler, BatchImage, BatchPrompt, CustomColdStartProxy, Image, Prompt

PROXY_URL = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchPrompt,
            output_type=BatchImage,
            cloud_build_config=L.BuildConfig(requirements=["flash-attn"]),
            *args,
            **kwargs,
        )

        self._requests = {}
        self._predictor_task = None
        self._lock = None

    def setup(self):
        if not os.path.exists("v1-5-pruned-emaonly.ckpt"):
            cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
            os.system(cmd)
        import ldm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="v1-5-pruned-emaonly.ckpt",
            device=device,
            fp16=True,  # Supported on GPU, skipped otherwise.
            deepspeed=True,  # Supported on Ampere and RTX, skipped otherwise.
            context="no_grad",
            flash_attention="hazy",
            steps=30,
        )

    def apply_model(self, requests):
        return self._model.in_loop_predict_step(requests)

    def sanetize_data(self, request):
        if "state" in request:
            return request["state"]
        return request["data"].text

    def sanetize_results(self, result):
        image = result
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return BatchImage(outputs=[{"image": image_str}])

    async def predict_fn(self):
        try:
            while True:
                async with self._lock:
                    keys = list(self._requests)

                if len(keys) == 0:
                    await asyncio.sleep(0.0001)
                    continue

                inputs = {key: self.sanetize_data(self._requests[key]) for key in keys}
                results = self.apply_model(inputs)

                for key, state in inputs.items():
                    if key == "global_state":
                        self._requests["global_state"] = {"state": state}
                    else:
                        self._requests[key]["state"] = state

                if results:
                    for key in results:
                        self._requests[key]["response"].set_result(self.sanetize_results(results[key]))
                        del self._requests[key]

                await asyncio.sleep(0.0001)
        except Exception:
            print(traceback.print_exc())

    async def predict(self, request: BatchPrompt):
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._predictor_task is None:
            self._predictor_task = asyncio.create_task(self.predict_fn())
        assert len(request.inputs) == 1
        future = asyncio.Future()
        async with self._lock:
            self._requests[uuid.uuid4().hex] = {"data": request.inputs[0], "response": future}
        result = await future
        return result


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu", disk_size=80),
    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=6,
    timeout_batching=0,
    input_type=Prompt,
    output_type=Image,
    batching="streamed",
)
# component = LanguageModelServer()
app = L.LightningApp(component)
