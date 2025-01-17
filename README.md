# BaoZaoAI
基于Qwen-2.5-1.5B-Instruct 进行DPO fine-tuning后，意外说真话的AI暴躁哥！
![](image.png)
## 使用说明和风险提示
这是一个基于LLM的聊天应用，旨在展示模型的对话生成能力。在使用前，请仔细阅读以下说明和风险提示。
## 用法说明

- 数据集处理(optional，可以直接用我上传的数据集)
    - 运行 1_prepare_data.ipynb
- DPO训练复现
    - 运行 2_DPO.ipynb
- 推理：
    - 运行demo.py文件，启动Streamlit应用。在浏览器中访问Streamlit提供的本地地址，开始与暴躁AI哥对话。
    - 输入问题：在聊天界面的输入框中输入你的问题或指令。
    - 接收回复：暴躁AI哥会以一种独特的、可能带有挑衅性的方式回复你的问题。
- 查看历史：你可以在聊天界面中查看之前的对话历史。

## 风险提示
- 脏话风险：请注意，这个模型在训练过程中使用了包含脏话的数据。因此，在使用过程中，模型可能会生成包含不雅语言的回复。我们强烈建议在公共场合或敏感环境中谨慎使用。
- 内容责任：虽然模型的回复可能具有挑衅性或不适当，但请记住，这些回复是由模型生成的，不代表开发者的观点或立场。用户应对自己的使用行为负责。
- 遵守法律法规：在使用过程中，请确保遵守当地的法律法规，尊重他人的权利和感受，模型未进行任何安全alignment，无法控制具体输出的内容，不要故意生成辱骂他人的内容。
- 政治敏感性：这是一个实验模型，未进行任何alignment安全性检查，请不要向模型提出任何政治相关的问题。
- 技术细节
使用DPO基于Qwen-2.5-1.5B-Instruct模型，基于脏话数据集进行fine-tuning。请注意，由于网络原因，上述链接可能无法直接访问。如果遇到链接无法访问的问题，建议检查网页链接的合法性，并适当重试。如果问题仍然存在，可能需要检查网络连接或联系Hugging Face平台获取帮助。
- 实验目的
这个模型仅用于测试DPO（Direct Preference Optimization）效果，它展示了模型在特定数据集上fine-tuning后可能表现出的意外行为。我们希望通过这个实验，能够更好地理解模型的行为，并在未来的开发中进行改进。
- 注意事项
- 实验性质：这是一个实验模型，可能存在不稳定或不可预测的行为。
- 安全性：由于模型未进行安全性检查，使用时请特别注意不要生成或传播不当内容。
- 反馈：我们鼓励用户提供反馈，以帮助我们改进模型的性能和安全性。
- 希望这个实验模型能够为您提供有趣的体验，同时也期待您的宝贵意见。请记住，使用时请遵守所有适用的法律法规，并尊重他人的权利和感受。