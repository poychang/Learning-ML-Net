using Microsoft.ML.Data;

namespace IrisPredictionTutorial.DataStructures
{
    // IrisData 資料模型用於訓練資料使用，並可作為預測資料模型
    //   - 前 4 個屬性為輸入的特性值，用來預測 Label 標籤
    //   - Label 標籤是我們要進行預測的屬性，只有在訓練資料時才會主動提供值
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
}
