import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-1d5bvZ7FBiHekMSUOtbATIVTjkfd7fBQfWnbU1VDxTmsm2ky",
    base_url="http://14.103.68.46/v1",
    timeout=10.0  # 核心：强制 10 秒超时，不准无限等！
)

print("正在呼叫测试...")
try:
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "你好，请回复数字 1"}],
        max_tokens=10
    )
    print("成功！响应内容：", response.choices[0].message.content)
except Exception as e:
    print("\n[实锤报错] 服务器无响应，真实原因：", e)