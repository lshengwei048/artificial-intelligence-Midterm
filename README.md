
# 人工智慧期中報告：如何使用Google Colab免費算力訓練自己的模型

# 組員:11224118林宗翰、11224123羅聖幃、11224127錢俊宇

## 一、專題簡介

本次人工智慧期中報告主要參考「如何使用 Google Colab 免費算力訓練自己的模型」這篇教學文章，學習如何使用 Google 提供的 Colab 雲端運算環境。

Google Colab 可以直接在瀏覽器中執行 Python 程式，不需要另外安裝完整的開發環境，也能夠使用雲端 GPU 進行運算。

本次實作主要包含 Google Drive 連線、Google Colab 基本操作、GPU 與 CPU 資訊查看、Python 套件安裝、資料下載及 GitHub 程式下載等功能。

透過這些操作，可以先熟悉 Google Colab 的開發環境，為未來進行人工智慧模型訓練做準備。

---

## 二、開發環境與工具

本次實作使用以下工具：

| 項目 | 使用工具 |
|---|---|
| 開發平台 | Google Colab |
| 程式語言 | Python 3 |
| 雲端儲存 | Google Drive |
| 深度學習套件 | TensorFlow、Keras |
| GPU 資訊查看 | NVIDIA-SMI |
| 系統環境 | Linux |
| 程式版本管理 | GitHub |

### Google Colab 的特色

1. 不需要在電腦中額外安裝 Python。
2. 可以透過瀏覽器直接執行程式。
3. 提供 GPU 運算資源，但免費額度有限。
4. 可以連接 Google Drive 儲存檔案。
5. 支援安裝 Python 套件。
6. 可以下載 GitHub 上的專案。

---

## 三、登入 Google Drive

首先開啟 Google Drive：

https://drive.google.com/drive/my-drive

登入自己的 Google 帳號後，進入「我的雲端硬碟」。

Google Drive 可以用來保存 Python 程式、資料集及其他檔案。

原教學文章是透過 Google Drive 安裝 Colaboratory 應用程式，再建立 Notebook。

現在也可以直接使用 Google Colab 網站，不需要先安裝額外的應用程式。

---

## 四、建立 Google Colab 筆記本

### 4.1 開啟 Google Colab

進入網站：

https://colab.research.google.com/

登入 Google 帳號後，選擇建立新的 Notebook。

Notebook 可以同時加入文字與 Python 程式碼，適合用來記錄實作過程。

### 4.2 Python 基本測試

在程式碼儲存格輸入：

```python
print("Hello Google Colab")
print("Google Colab 環境測試成功")
```

執行結果：

```text
Hello Google Colab
Google Colab 環境測試成功
```

透過這個測試，可以確認 Google Colab 能夠正常執行 Python 程式。

---

## 五、連接 Google Drive

原本教學文章使用 google-drive-ocamlfuse 套件掛載 Google Drive。

由於文章中的部分安裝方式已經過時，本次改用 Google Colab 內建的 Google Drive 掛載功能。

### 5.1 掛載 Google Drive

執行以下程式：

```python
from google.colab import drive

drive.mount("/content/drive")
```

執行後依照畫面指示授權 Google Drive。

成功後會顯示：

```text
Mounted at /content/drive
```

### 5.2 查看雲端硬碟資料

```python
!ls /content/drive/MyDrive
```

如果成功列出 Google Drive 中的資料，就代表連接正常。

### 5.3 建立資料夾

```python
import os

folder = "/content/drive/MyDrive/Colab_Project"

os.makedirs(folder, exist_ok=True)

print("資料夾建立完成")
```

### 5.4 儲存測試檔案

```python
file_path = "/content/drive/MyDrive/Colab_Project/test.txt"

with open(file_path, "w", encoding="utf-8") as f:
    f.write("Google Colab 測試成功")

print("檔案已儲存")
```

執行完成後，可以前往 Google Drive 查看剛剛建立的檔案。

---

## 六、GPU 運算環境檢查

Google Colab 可以依照可用資源提供 GPU。

GPU 可以加速部分大量運算的工作，常用於人工智慧及深度學習。

### 6.1 啟用 GPU

在 Google Colab 上方選擇：

`執行階段 → 變更執行階段類型 → 硬體加速器 → GPU`

儲存後重新連接執行階段。

### 6.2 查看 GPU 是否可用

原教學使用 TensorFlow 檢查 GPU，本次使用以下程式：

```python
import tensorflow as tf

print("TensorFlow 版本：", tf.__version__)

gpu_devices = tf.config.list_physical_devices("GPU")

print("GPU 裝置：", gpu_devices)

if gpu_devices:
    print("GPU 可以正常使用")
else:
    print("目前沒有可使用的 GPU")
```

如果成功取得 GPU，程式就會顯示裝置資訊。

如果沒有 GPU，則可以繼續使用 CPU 執行一般 Python 程式。

---

## 七、查看 GPU 詳細資訊

除了使用 TensorFlow 檢查 GPU，也可以使用 NVIDIA-SMI 查看顯示卡資訊。

### 7.1 查看 GPU 狀態

```python
!nvidia-smi
```

執行後，如果目前有 NVIDIA GPU，通常可以看到：

- GPU 型號
- GPU 記憶體容量
- GPU 記憶體使用量
- GPU 使用率
- CUDA 驅動資訊

實際顯示的 GPU 型號與容量，會依 Colab 當次分配的硬體而不同。

### 7.2 查看 TensorFlow 裝置

```python
from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()

for device in devices:
    print(device)
```

這段程式可以列出 TensorFlow 能夠使用的本機運算裝置。

透過 GPU 資訊查看，可以知道目前的雲端運算環境是否適合後續進行模型訓練。

---

## 八、Python 套件安裝與移除

Google Colab 可以使用 pip 指令安裝或更新 Python 套件。

原教學以 Keras 作為安裝與移除的範例。

### 8.1 安裝 Keras

```python
%pip install -q keras
```

### 8.2 匯入 Keras

```python
import keras

print("Keras 版本：", keras.__version__)
```

如果程式正常顯示版本資訊，代表 Keras 可以使用。

### 8.3 移除套件

如果確定不再需要某個套件，可以使用：

```python
%pip uninstall -y 套件名稱
```

其中 `-y` 代表自動確認移除。

例如：

```python
%pip uninstall -y example-package
```

這裡的 example-package 是示範用名稱，實際使用時須替換為已安裝的套件。

注意：Keras 可能是 TensorFlow 環境的重要相依套件，不建議為了測試而直接移除。

---

## 九、Google Colab 工作目錄操作

在 Colab 中執行程式時，需要了解目前檔案所在的位置。

### 9.1 查看目前目錄

```python
!pwd
```

### 9.2 查看目錄內容

```python
!ls
```

### 9.3 使用 Python 切換目錄

```python
import os

print("切換前：", os.getcwd())

os.chdir("/content")

print("切換後：", os.getcwd())
```

### 9.4 使用 Colab 指令切換目錄

```python
%cd /content
```

### 9.5 切換至 Google Drive

需要先完成第五章的 Google Drive 掛載。

```python
%cd /content/drive/MyDrive
```

再執行：

```python
!pwd
!ls
```

這樣就能確認目前是否已經進入 Google Drive。

需要注意的是，使用 `!cd` 執行的 shell 切換不會持續影響 Notebook 後續儲存格的工作目錄，因此通常建議使用 `%cd` 或 `os.chdir()`。

---

## 十、使用 GitHub 下載專案

Google Colab 也支援 Git 指令，可以將 GitHub 上的公開專案下載至雲端環境。

### 10.1 使用 git clone

原教學文章提供以下 GitHub 專案作為範例：

```python
!git clone https://github.com/wxs/keras-mnist-tutorial.git
```

這個指令會嘗試將遠端 GitHub 專案複製到目前的工作目錄。

### 10.2 查看下載的檔案

```python
!ls
```

也可以進入下載完成的專案資料夾：

```python
%cd keras-mnist-tutorial
```

### 10.3 查看專案內容

```python
!ls -la
```

透過 GitHub 下載功能，可以在 Colab 中使用其他開發者提供的公開程式碼。

本次僅示範下載與查看檔案，沒有執行該專案的模型訓練。

---

## 十一、資料下載與系統資訊查看

原教學也介紹如何下載網路上的資料，以及查看 Colab 的 CPU 和記憶體資訊。

### 11.1 使用 wget 下載資料

原文使用 Titanic CSV 檔案作為下載範例：

```python
!wget https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Titanic.csv
```

下載完成後，可以使用：

```python
!ls
```

查看目前資料夾中是否有下載完成的 CSV 檔案。

### 11.2 指定下載位置

以下範例會先建立目錄，再將檔案下載到指定位置：

```python
import os

os.makedirs("/content/data", exist_ok=True)
```

```python
!wget -P /content/data https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Titanic.csv
```

### 11.3 查看系統記憶體資訊

```python
!cat /proc/meminfo
```

執行後可以查看目前系統的記憶體資訊。

### 11.4 查看 CPU 資訊

```python
!cat /proc/cpuinfo
```

執行後可以查看目前的 CPU 處理器資訊。

### 11.5 查看磁碟空間

```python
!df -h
```

這段指令可以查看 Colab 執行環境目前的磁碟容量與使用情況。

透過這些 Linux 指令，可以進一步了解 Google Colab 雲端主機的硬體與系統環境。

---

## 十二、結論與心得

透過這次 Google Colab 的實作，我們對雲端 Python 開發環境有更進一步的認識。

原本要執行人工智慧相關程式，可能需要在電腦上安裝 Python、TensorFlow 等套件，還要考慮電腦的硬體效能。

使用 Google Colab 後，可以直接透過瀏覽器執行 Python，也能依照資源狀況使用 Google 提供的 GPU。

在這次操作中，我們學習了如何連接 Google Drive、查看 GPU 與 CPU 資訊、安裝 Python 套件，以及使用 Linux 指令管理檔案。

另外，也了解如何透過 GitHub 下載程式碼，以及使用 wget 下載資料。

這些功能讓我了解到 Google Colab 不只是能夠執行 Python 的工具，也是一個適合學習人工智慧與進行程式實驗的雲端平台。

雖然這次參考文章主要介紹環境設定，尚未提供完整的模型訓練實作，但透過這些基礎操作，已經能夠先建立後續進行人工智慧學習所需要的開發環境。

### 本次實作項目整理

| 測試項目 | 使用指令或工具 |
|---|---|
| 建立 Colab Notebook | Google Colab |
| Google Drive 連接 | drive.mount() |
| GPU 可用性檢查 | TensorFlow |
| GPU 資訊查看 | nvidia-smi |
| Python 套件安裝 | pip install |
| Python 套件移除 | pip uninstall |
| 查看工作目錄 | pwd |
| 切換工作目錄 | %cd |
| 查看資料夾內容 | ls |
| 下載 GitHub 專案 | git clone |
| 下載網路資料 | wget |
| 查看記憶體資訊 | /proc/meminfo |
| 查看 CPU 資訊 | /proc/cpuinfo |

---

**人工智慧期中報告 — Google Colab 免費雲端運算環境實作**
