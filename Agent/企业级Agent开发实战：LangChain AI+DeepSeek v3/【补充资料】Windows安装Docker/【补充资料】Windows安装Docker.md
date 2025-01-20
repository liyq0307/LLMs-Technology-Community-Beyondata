## 课程说明：

* 体验课内容节选⾃[《2025⼤模型Agent智能体开发实战》](https://appze9inzwc2314.h5.xiaoeknow.com/v1/goods/goods_detail/p_67472baee4b0694c3c6fc88a?type=3\&channel_id=)完整版付费课程

体验课时间有限，若想深度学习⼤模型技术，欢迎⼤家报名由我主讲的[《2025⼤模型Agent智能体开发实战》](https://appze9inzwc2314.h5.xiaoeknow.com/v1/goods/goods_detail/p_67472baee4b0694c3c6fc88a?type=3\&channel_id=)：

![](images/image.png)

![](images/f339b04b7b20233dd1509c7fb36d5c0.png)

此外，公开课全部课件，以及项目代码、数据等学习资料，扫码⬆️联系助教即可领取～

***

# 1. 安装Git

* **Step 1. 进入Git官网：https://git-scm.com/**

![](images/0712c484-104d-4359-b415-ba0fc21b7631.png)

* **Step 2. 下载Windwos版本的Git安装程序**

![](images/00b5242a-6eec-4b7e-b5f0-52aeb9c2b582.png)

  这里选择Windwos版本进行下载。

![](images/dfe5ad2e-6755-4fd6-b6d5-e8dbb2dd571c.png)

  根据自己的Windwos系统情况，选择32-bit或者64bit。

![](images/9591bdf5-98d8-4bf3-bdf5-676e83813d81.png)

  选择Git安装程序的存储路径。

![](images/750a7b8c-28e4-4349-a756-d55ad1207f8c.png)

* **Step 3. 安装Git**

![](images/22a2de8b-9d12-40b7-b74a-d24050e6be73.png)

  选择Next。

![](images/3e97054f-c5c8-422a-8526-d6997a54eec3.png)

  自行选择Git的安装位置。

![](images/c013786e-cb84-4b45-a0cd-414b0d8e759d.png)

  接下来的选项，全部默认，点击Next即可。不影响正常的使用。

![](images/1ac9cc22-3ead-4426-99db-f03c42110b15.png)

![](images/2ff75bf4-9860-447e-9716-8a9b9503d646.png)

![](images/73ead65e-9ecb-4bc5-996c-b7047cf78a48.png)

![](images/fe889c12-64be-4abd-955a-9cce4190b721.png)

![](images/d4cebcbd-40be-4e91-ba92-b09d1384f7f4.png)

![](images/4972f8f9-f005-41c4-8027-a80d189874ac.png)

![](images/a9748420-5d37-471d-8f53-c6ea31d420c3.png)

![](images/d7c641e2-0afa-4edf-9063-f6443e1f03a8.png)

![](images/4a9c642c-b9c7-4fcf-97b5-8342449074d3.png)

![](images/d6e44e17-b4b2-4e94-9f07-36209466ac9c.png)

![](images/38678a7b-105c-438d-839c-edcd67240bd8.png)

![](images/c5aa0c6c-8204-4f78-96cd-ee992a70cb85.png)

  默认选择到这里，执行安装。

![](images/978b0a4e-5d55-4ba5-9823-eab5e56c2fc2.png)

  执行到此进度，说明安装已经完成。

![](images/6b4c6bce-2c12-400d-9895-c3005c10770e.png)

* **Step 4. 验证Git安装是否成功**

  在桌面上鼠标右键，如能看到如下两个快捷方式，说明已经正常安装Git。

![](images/8baf1d70-0b63-4fae-b271-90d041a99576.png)

  点击Git Bash 后，进入命令行终端，可以输入`git --version` 查看当前的Git版本。

![](images/44662c4f-9ede-4b1e-a255-58564fcad46b.png)

# 2. 安装 FFmpeg

  下面是如何在Windows系统上安装和配置`ffmpeg`的步骤：

1. **下载FFmpeg**：访问[BtbN's Windows builds on GitHub](https://github.com/BtbN/FFmpeg-Builds/releases)，下载适用于Windows的FFmpeg压缩文件。

![](images/bc1a9be2-0f90-4ab1-b4d2-78e8bc918def.png)

2. **解压FFmpeg**：将下载的文件解压到你选择的目录，比如`C:\FFmpeg`。

3. **配置环境变量**：

   * 右键点击“此电脑”或“我的电脑”，选择“属性”。

   * 点击“高级系统设置”，然后选择“环境变量”。

   * 在“系统变量”部分，找到并选择`Path`变量，然后点击“编辑”。

   * 点击“新建”，然后添加FFmpeg的bin目录的路径，比如`C:\FFmpeg\bin`。

   * 点击“确定”保存你的设置。

4. **验证安装**：

   * 打开命令提示符（CMD），输入`ffmpeg -version`并回车。如果系统返回了FFmpeg的版本信息，说明FFmpeg已经成功安装和配置。

![](images/0fb41576-42fe-4840-952b-22c79674cc1d.png)

# 3. Window 安装 Docker

  Docker 是一个开源的平台，它可以使软件开发和部署变得更加简便和高效。通过使用 Docker，开发者可以将应用程序及其所有依赖项打包到一个被称为“容器”的轻量级、可移植的、自给自足的单元中。这样，无论在哪种操作系统或环境下，这个容器都能保证以相同的方式运行。

  想象一下，你把一个应用程序比作一个需要特定环境才能生存的植物。如果你想把这个植物带到不同的地方（不同的计算环境），你需要确保每个地方都有合适的土壤、光照和水分。Docker 就像是一个可以随处携带的植物生存盒，无论你把它放到哪里，它都包含了植物生长所需要的一切条件。这样，你就不需要担心不同地方的环境变化，植物（应用程序）总是可以顺利成长。

  这种方式使得软件部署更加简单，也减少了因环境不一致导致的问题，这在开发和测试阶段特别有帮助。因此，Docker 在云计算和微服务架构中非常受欢迎。

  Windows Desktop桌面版官方下载地址：https://www.docker.com/products/docker-desktop/

* **Step 1. 下载Windwos安装程序**

![](images/e1e11692-ec13-4bce-9d54-712e96ae6a0e.png)

  选择Windows操作系统版本。

![](images/a498aae8-5aa5-414c-9f20-932f06bf2214.png)

  选择安装包存放路径，点击等待下载完成即可。

![](images/ff09a917-561d-4e93-ab40-4e07c127c29f.png)

* **Step 2. 安装完双击打开Docker**

![](images/a98af805-357c-4653-837e-9fb9e461790c.png)

* **Step 3. 执行安装过程**

![](images/cf36987f-f0c0-4973-9db9-32b21219ff5f.png)

![](images/1f95af9f-933a-4426-9822-92ca1f381e68.png)

* **Step 4. 安装完成后，需要重启电脑才能加载Docker应用**

![](images/ce5f644e-8e8f-41b0-96d4-abc382f50d48.png)

* **Step 5. 重启完电脑后，打开Docker Desktop客户端，首次使用需要接受服务协议**

![](images/1f37c957-c1ac-442b-b918-52fc29e2a2dd.png)

* **Step 6. 这里选择默认配置即可**

![](images/e5110647-3e23-4c90-b7ba-510b670166e5.png)

* **Step 7. 在使用前，需要登陆账户，可以免费注册**

![](images/c1471fbe-0351-4967-b79c-f55a40d2bbe3.png)

  登录后，即可正常使用Docker服务了。

![](images/ea76800c-6f2f-4fc8-86d8-634dd3b58772.png)
