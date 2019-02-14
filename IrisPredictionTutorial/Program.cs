using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace IrisPredictionTutorial
{
    public class Program
    {
        // STEP 1: 定義資料模型
        //         IrisData 資料模型用於訓練資料使用，並可作為預測資料模型
        //         - 前 4 個屬性為輸入的特性值，用來預測 Label 標籤
        //         - Label 標籤是我們要進行預測的屬性，只有在訓練資料時才會主動提供值
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        // IrisPrediction 是執行預測所產生的結果資料模型
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        public static void Main(string[] args)
        {
            // STEP 2: 建立執行 ML.NET 的執行環境
            var mlContext = new MLContext();

            // 注意！若使用 Visual Studio 開發，請確認 iris-data.txt 檔案有將"複製到輸出目錄" (Copy to Output Directory) 屬性設定成"一律複製" (Copy always)
            // 資料集來源：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
            var trainingDataView = mlContext.Data.ReadFromTextFile<IrisData>("iris-data.txt", false, ',');

            // STEP 3: 轉換資料並加入學習器
            // 將訓練資料集的 Label 文字欄位轉成數值，因為只有數值能放進訓練預測模型的程序中
            // 將適合的學習演算法加入學習管線 (pipeline) 中，例如我們要預測該鳶尾花的類型
            // 完成訓練後，將稍早轉成數值的 Label 欄位再轉成原來的文字
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: 基於所提供的訓練資料集進行訓練，將模型定型
            var model = pipeline.Fit(trainingDataView);

            // STEP 5: 使用訓練好的預測模型檔進行預測
            // 你可以修改下列數值，測試你所定型的模型
            var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();
        }
    }
}
