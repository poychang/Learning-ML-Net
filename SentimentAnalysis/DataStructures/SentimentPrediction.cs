using Microsoft.ML.Data;

namespace SentimentAnalysis.DataStructures
{
    public class SentimentPrediction
    {
        // ColumnName 屬性用於更改列名稱的預設值，即字段的名稱。
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // 如果名稱是我們想要的，則不用透過設定 ColumnName 屬性
        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
