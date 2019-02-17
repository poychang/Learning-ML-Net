using IrisPredictionTutorial.DataStructures;
using Microsoft.ML;
using System;

namespace IrisPredictionTutorial
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // STEP 1: 定義資料模型 (Data Structure)
            // 參考 DataStructure 資料夾的程式碼

            // STEP 2: 建立執行 ML.NET 的執行環境 (Create ML Context)
            var mlContext = new MLContext();

            // STEP 3: 載入訓練資料集 (Load Datasets)
            // 注意！若使用 Visual Studio 開發，請確認 iris-data.txt 檔案有將"複製到輸出目錄" (Copy to Output Directory) 屬性設定成"一律複製" (Copy always)
            // 資料集來源：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
            var trainingDataView = mlContext.Data.ReadFromTextFile<IrisData>("Data/iris-data.txt", false, ',');

            // STEP 4: 建立學習管線 (Build Pipeline)
            // 轉換資料，將訓練資料集的 Label 文字欄位轉成數值，因為只有數值能放進訓練預測模型的程序中
            // 加入學習器，將適合的學習演算法(StochasticDualCoordinateAscent)加入學習管線中
            // 完成訓練後，將稍早轉成數值的 Label 欄位再轉成原來的文字
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 5: 模型定型 (Fit Model)
            // 基於所提供的訓練資料集進行訓練，將模型定型
            var model = pipeline.Fit(trainingDataView);

            // STEP 6: 驗證模型 (Verify Model)
            // 使用訓練好的預測模型檔對測試資料集進行預測
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
