using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.MNIST28
{
    /*
     * Data Source: https://github.com/primaryobjects/DigitRecognizer/tree/master/data train.csv
     * Data model class for multi classification
     */
    public class MNIST28DataModel
    {
        [ColumnName("Pixels")]
        [VectorType(784)]
        public float[] Pixels;

        [LoadColumn(784)]
        public float Digit;               
    }
}
