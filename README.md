# Learning ML.NET

## 機器學習任務類型

- 二元分類 Binary classification
  - 將提供的資料集分類成兩組，並預測資料是屬於哪一組
- 多元分類 Multi-class classification
  - 將提供的資料及分類成多組，並預測資料是屬於哪一組
- 回歸 Regression
  - 利用所提供的資料集建立一組可以表示該資料集的函數，常用於預測趨勢
- 推薦 Recommendation
- 集群 Clustering
  - 探索資料集中的群聚，將彼此相似的數據做歸類
- 異常檢測 Anomaly detection
- 排名 Ranking
- 深度學習 Deep Learning

## 機器學習專案的基本流程

STEP 0: 準備資料
  - 取得學習資料集，建議將資料分成三類
    - 全部資料集，如 `iris-all.txt`
    - 訓練資料集，如 `iris-data.txt`
    - 驗證資料集，如 `iris-test.txt`

STEP 1: 定義資料模型 (Data Structure)
  - 會建立兩種資料模型，分別用於訓練及預測
    - 訓練資料模型，輸入，通常和資料的欄位結構一致
    - 預測資料模型，輸出，預測的結果會用此模型來呈現

STEP 2: 建立執行 ML.NET 的執行環境 

STEP 3: 載入訓練資料集 (Load Datasets)
    - 注視資料格式
    - 轉換資料
      - 執行訓練的過程中，只有數值能被計算，因此**若資料有文字類型的欄位，要先轉換成數值**
      - 訓練完成後再轉換成原本的文字

STEP 4: 建立學習管線 (Build Pipeline)

STEP 5: 模型定型 (Fit Model)

STEP 6: 驗證模型 (Verify Model)
  - 務必在一開始保留部分原始資料，作為驗證資料集，避免過擬的狀態發生

## Note

一個完整的機器學習專案，可以概括為如下步驟：

1. 數據清洗
   - 缺失值
   - 數據類型
   - 異常值
   - 文本編碼
   - 數據分割
2. 特徵提取
   - 數值延伸特徵
   - 離散特徵
   - 文本特徵
   - 時序特徵
   - 交叉特徵
3. 特徵選擇、降維
   - 線性投影
   - 非線性投影
   - 特徵篩選
4. 模型選擇、訓練
   - 模型選擇
   - 參數選擇
   - 模型訓練

## 參考資料

- [dotnet/machinelearning-samples](https://github.com/dotnet/machinelearning-samples)
- [Google 機器學習術語庫](https://developers.google.cn/machine-learning/glossary/?hl=zh-CN)
