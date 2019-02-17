using Microsoft.ML.Data;

namespace SentimentAnalysis.DataStructures
{
    public class SentimentIssue
    {
        // LoadColumn 屬性用來設定資料來源是在哪一欄
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
