from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Prompt(BaseModel):
    text: Optional[str]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        return {"text": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"}

    @staticmethod
    def request_code_sample(url: str) -> str:
        return (
            """import base64
from pathlib import Path
import requests

response = requests.post('"""
            + url
            + """', json={
"text": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"
})
# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """', json={
# "text": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"
# }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))
"""
        )


class BatchPrompt(BaseModel):
    # Note: field name must be `inputs`
    inputs: List[Prompt]

    @staticmethod
    def request_code_sample(url: str) -> str:
        return (
            """import requests
response = requests.post('"""
            + url
            + """', json={
"inputs": [{"text": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"}]
})
# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """', json={
# "inputs": [{"text": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:"}],
# }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))
"""
        )

class ModelOutput(BaseModel):
    text: Optional[str]
    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        return {"text": """        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [dic[nums[i]], i]
            dic[target - nums[i]] = i
        return []"""}

class BatchModelOutput(BaseModel):
    # Note: field name must be `outputs`
    outputs: List[ModelOutput]

    @staticmethod
    def response_code_sample() -> str:
        return """text = response.json()["outputs"][0]["text"]"""