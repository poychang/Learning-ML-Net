using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace WineQuality
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), @"Data\winequality-red.csv");
            //var dataPath = Path.Combine(Directory.GetCurrentDirectory(), @"Data\winequality-white.csv");
            var ml = new MLContext();
            var DataView = ml.Data.LoadFromTextFile<InputDataModel>(dataPath, hasHeader: true, separatorChar: ';');

            // 設定 30% 為測試集
            var partitions = ml.Data.TrainTestSplit(DataView, testFraction: 0.3);
            var pipeline = ml.Transforms.Conversion.MapValueToKey(inputColumnName: "Quality", outputColumnName: "Label")
                .Append(ml.Transforms.Concatenate("Features", "FixedAcidity", "VolatileAcidity", "CitricAcid", "ResidualSugar", "Chlorides", "FreeSulfurDioxide", "TotalSulfurDioxide", "Density", "Ph", "Sulphates", "Alcohol"))
                .AppendCacheCheckpoint(ml);

            var trainingPipeline = pipeline
                .Append(ml.MulticlassClassification.Trainers.NaiveBayes("Label", "Features"))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = trainingPipeline.Fit(partitions.TrainSet);
            var testMetrics = ml.MulticlassClassification.Evaluate(trainedModel.Transform(partitions.TestSet));

            Console.WriteLine("Multi-Class Classification Algorithm: NaiveBayes");
            Console.WriteLine("Assessing MLmode1 Quality using Evaluation Metrics");
            Console.WriteLine();
            Console.WriteLine("----------------------------");
            Console.WriteLine("Confusion Matrix: {metrics.ConfusionMatrix.GetFormattedConfusionTab1e().ToString()}");
            Console.WriteLine();
            Console.WriteLine($"Macro Accuracy: {testMetrics.MacroAccuracy:P2}");
            Console.WriteLine($"Micro Accuracy: {testMetrics.MicroAccuracy:P2}");
            Console.WriteLine();
            Console.WriteLine($"Log Loss: {testMetrics.LogLoss}");
            Console.WriteLine($"Log Loss Reduction: {testMetrics.LogLossReduction}");
            Console.WriteLine();
            Console.WriteLine($"Top K Accuracy: {testMetrics.TopKAccuracy:P2}");
            Console.WriteLine($"Top K Prediction Count: {testMetrics.TopKPredictionCount}");
            Console.WriteLine("----------------------------");
            Console.WriteLine("End of Model Quality Metrics");
            Console.WriteLine("----------------------------");
        }
    }

    public class InputDataModel
    {
        [LoadColumn(0)]
        public float FixedAcidity { get; set; }
        [LoadColumn(1)]
        public float VolatileAcidity { get; set; }
        [LoadColumn(2)]
        public float CitricAcid { get; set; }
        [LoadColumn(3)]
        public float ResidualSugar { get; set; }
        [LoadColumn(4)]
        public float Chlorides { get; set; }
        [LoadColumn(5)]
        public float FreeSulfurDioxide { get; set; }
        [LoadColumn(6)]
        public float TotalSulfurDioxide { get; set; }
        [LoadColumn(7)]
        public float Density { get; set; }
        [LoadColumn(8)]
        public float Ph { get; set; }
        [LoadColumn(9)]
        public float Sulphates { get; set; }
        [LoadColumn(10)]
        public float Alcohol { get; set; }
        [LoadColumn(11)]
        public int Quality { get; set; }
    }

    public class OutputDataModel
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
