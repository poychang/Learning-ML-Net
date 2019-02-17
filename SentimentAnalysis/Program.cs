using Common;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using SentimentAnalysis.DataStructures;
using System;
using System.IO;

namespace SentimentAnalysis
{
    public class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private const string TrainDataPath = "Data/wikipedia-detox-250-line-data.tsv";
        private const string TestDataPath = "Data/wikipedia-detox-250-line-test.tsv";
        private const string ModelPath = "MLModels/SentimentModel.zip";

        public static void Main(string[] args)
        {
            // STEP 1: 定義資料模型
            // 參考 DataStructure 資料夾的程式碼

            // STEP 2: 建立執行 ML.NET 的執行環境
            // 為了讓整個建立模型的工作流程共用 MLContext，建立執行 ML.NET 的執行環境時給一個亂數種子，作為重複或展示使用的訓練環境
            // 請注意，這作法通常用於非正式環境中
            var mlContext = new MLContext(seed: 1);

            // STEP 3: 轉換資料，加入學習器
            // STEP 4: 基於所提供的訓練資料集進行訓練，將模型定型
            BuildTrainEvaluateAndSaveModel(mlContext);
            Common.ConsoleHelper.ConsoleWriteHeader("=============== End of training process ===============");

            // STEP 5: 使用訓練好的預測模型檔進行預測
            TestSinglePrediction(mlContext);

            Common.ConsoleHelper.ConsoleWriteHeader("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: 載入訓練資料集
            var trainingDataView = mlContext.Data.ReadFromTextFile<SentimentIssue>(TrainDataPath, hasHeader: true);

            // STEP 2: 建立學習管線
            // 這裡的資料來源是句子，所以使用 FeaturizeText 將資料轉換成數值向量，進行特徵化
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentIssue.Text));

            // (選用) 挑選 2 筆訓練資料集中的資料，並查看經過學習管線的特徵化後的結果
            ConsoleHelper.PeekDataViewInConsole<SentimentIssue>(mlContext, trainingDataView, dataProcessPipeline, 2);
            ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, DefaultColumnNames.Features, trainingDataView, dataProcessPipeline, 1);

            // STEP 3: 加入學習器
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: 基於所提供的訓練資料集進行訓練，將模型定型
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // STEP 5: 使用訓練好的預測模型檔對測試資料集進行預測
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var testDataView = mlContext.Data.ReadFromTextFile<SentimentIssue>(TestDataPath, hasHeader: true);
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, label: DefaultColumnNames.Label, score: DefaultColumnNames.Score);

            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            // STEP 6: 將定型的模型儲存成 .ZIP file 提供其他應用程式使用
            Directory.CreateDirectory(Path.GetDirectoryName(ModelPath));
            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(trainedModel, fs);
                Console.WriteLine("The model is saved to {0}", ModelPath);
            }

            return trainedModel;
        }

        // (選用) 使用儲存的定型模型進行預測
        private static void TestSinglePrediction(MLContext mlContext)
        {
            // 要預測的資料來源
            var sampleStatement = new SentimentIssue { Text = "This is a very rude movie" };
            // 載入定型模型
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var trainedModel = mlContext.Model.Load(stream);
                // 建立預測引擎
                var predictionEngine = trainedModel.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(mlContext);
                // 計算預測值
                var predict = predictionEngine.Predict(sampleStatement);

                Console.WriteLine($"=============== Single Prediction  ===============");
                Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(predict.Prediction) ? "Toxic" : "Nice")} sentiment | Probability: {predict.Probability} ");
                Console.WriteLine($"==================================================");
            }
        }
    }
}
