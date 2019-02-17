using Microsoft.ML.Data;

namespace IrisPredictionTutorial.DataStructures
{
    // IrisPrediction 是執行預測所產生的結果資料模型
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
