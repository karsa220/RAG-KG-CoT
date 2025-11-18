from zhipuai import ZhipuAI
import os
client = ZhipuAI(api_key="1326ff57993045e49f5da33f976983c6.TirEy4knJh50KckM")

response = client.chat.completions.create(
    model="glm-4-0520",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message["content"])
