## 通过API-Key调用Gemini模型

* API调用流程测试

```python
import google.generativeai as genai
from google.generativeai import GenerativeModel
```

```python
genai.configure(api_key='AIzaSyDX4AaXpLtsCFMmQSIdmnweCVp8Npbulf4')
```

```python
model = GenerativeModel('gemini-pro')
response = model.generate_content('The opposite of hot is')
print(response.text)
```

```plaintext
cold
```

* 多轮对话测试

```python
from IPython.display import Markdown
from IPython.display import display
import pathlib
import textwrap

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
```

```python
chat = model.start_chat()
```

```python
to_markdown(chat.send_message("请问什么是机器学习？").text)
```

* \*\*监督学习（Supervised Learning）：\*\*在监督学习中，计算机使用标记的数据（即包含正确答案的数据）来训练模型。然后，模型可以使用新数据来做出预测。

* \*\*无监督学习（Unsupervised Learning）：\*\*在无监督学习中，计算机使用未标记的数据来训练模型。模型必须自己找出数据中的模式，然后使用这些模式来做出预测。

* \*\*图像识别（Image Recognition）：\*\*计算机学习识别图像中的物体。

* \*\*自然语言处理（Natural Language Processing）：\*\*计算机学习理解和生成人类语言。

* \*\*语音识别（Speech Recognition）：\*\*计算机学习理解人类的语音。

* \*\*推荐系统（Recommendation Systems）：\*\*计算机学习向用户推荐产品或服务。

* \*\*欺诈检测（Fraud Detection）：\*\*计算机学习识别欺诈性交易。

* \*\*医学诊断（Medical Diagnosis）：\*\*计算机学习识别疾病，并推荐治疗方案。

```python
chat.send_message("我的上一个问题是？").text
```

```plaintext
'您的上一个问题是：“机器学习是什么？”\n\n您问这个问题是为了了解机器学习的基本概念和应用。我给出了一个全面的回答，解释了机器学习的定义、类型、算法和应用领域。希望我的回答对您有所帮助。\n\n如果您还有其他问题，请随时问我。'
```

* 多模态功能测试

```python
import PIL.Image
image = PIL.Image.open('image.jpg')
```

```python
image
```

![](images/default.png)

```python
multimodal_model = GenerativeModel('gemini-pro-vision')
response = multimodal_model.generate_content(['请帮我介绍下图片上展示的内容', image])
print(response.text)
```

```plaintext
 图片上展示了一只可爱的小猫在草地上奔跑。小猫全身都是蓬松的毛发，大大的眼睛炯炯有神，尾巴高高的翘起，看起来非常高兴。它在草地上奔跑，享受着自由自在的感觉。
```

## 通过Vertex AI调用Gemini模型

```python
# 安装google-cloud-aiplatform
!pip install google-cloud-aiplatform
```

```python
# 导入生成式AI库
from vertexai.preview.generative_models import GenerativeModel, Image, Content, Part, Tool, FunctionDeclaration, GenerationConfig, HarmCategory, HarmBlockThreshold
```

```python
from google.cloud import aiplatform
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import vertexai
#from vertexai.generative_models import GenerativeModel, Part

# 从本地文件中加载凭据
creds = Credentials.from_authorized_user_file('token.json')
```

```python
aiplatform.init(credentials=creds, project='rock-terra-407914')
```

* 文本对话测试

```python
model = GenerativeModel("gemini-pro")
```

```python
response = model.generate_content('The opposite of hot is')
print(response.text)
```

```plaintext
Cold
```

* 多轮对话功能测试

```python
from IPython.display import Markdown
from IPython.display import display
import pathlib
import textwrap

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
```

```python
model = GenerativeModel('gemini-pro')
chat = model.start_chat()
```

```python
to_markdown(chat.send_message("请问什么是深度学习？").text)
```

* 图像识别：深度学习模型可以用于识别图像中的物体。这对于人脸识别、物体检测和自动驾驶等任务非常有用。

* 自然语言处理：深度学习模型可以用于处理自然语言，例如翻译语言、生成文本和回答问题。

* 语音识别：深度学习模型可以用于识别语音，例如语音转文本和语音控制。

* 医学成像：深度学习模型可以用于分析医学图像，例如检测癌症和诊断疾病。

* 金融欺诈检测：深度学习模型可以用于检测金融欺诈，例如信用卡欺诈和洗钱。

* 药物发现：深度学习模型可以用于发现新药，例如设计新分子和预测药物的副作用。

```python
to_markdown(chat.send_message("请问我的上一个问题是？").text)
```

* 多模态性能测试

```python
# 导入图片
image = Image.load_from_file("image.jpg")
```

```python
image
```

![](images/default-1.png)

```python
# 创建模型
vision_model = GenerativeModel("gemini-pro-vision")
```

```python
vision_model.generate_content(["请帮我介绍下图片上展示的内容", image]).text
```

```plaintext
' 图片上展示了一只可爱的小猫在草地上奔跑。小猫全身毛茸茸的，眼睛是蓝色的，非常可爱。它正在追逐一只蝴蝶。小猫跑得很开心，它的心情看起来很好。'
```



📍**更多大模型技术内容学习**

**扫码添加助理英英，回复“大模型”，了解更多大模型技术详情哦👇**

![](images/f339b04b7b20233dd1509c7fb36d5c0.png)

此外，**扫码回复“入群”**，即可加入**大模型技术社群：海量硬核独家技术`干货内容`+无门槛`技术交流`！**
