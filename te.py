
from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="1326ff57993045e49f5da33f976983c6.TirEy4knJh50KckM")  # 请填写您自己的 API Key

response = client.chat.completions.create(
    model="glm-4.5-flash",
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
        {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
        {"role": "user", "content": "智谱AI 开放平台"}
    ],
    thinking={
        "type": "enabled",    # 启用深度思考模式
    },
    stream=True,              # 启用流式输出
    max_tokens=4096,          # 最大输出 tokens
    temperature=0.7           # 控制输出的随机性
)

# 获取回复
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')