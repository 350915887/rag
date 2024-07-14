from langchain_openai import ChatOpenAI
import jwt
import time
from langchain_core.messages import HumanMessage

zhipuai_api_key = "79fc446d4f1d1c531a4f6a984b11439c.02E4szGhFRXgJNgC"


def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


class ChatZhiPuAI(ChatOpenAI):
    def __init__(self, model_name):
        super().__init__(model_name=model_name, openai_api_key=generate_token(zhipuai_api_key, 10),
                         openai_api_base="https://open.bigmodel.cn/api/paas/v4")

    def invoke(self, question):
        messages = [
            HumanMessage(content=question),
        ]
        return super().invoke(messages)

