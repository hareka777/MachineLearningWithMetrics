using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.MNIST28
{
    public class MNIST28DataModel
    {
        [ColumnName("Pixels")]
        [VectorType(784)]
        public float[] Pixels;

        [LoadColumn(784)]
        public float Digit;               
    }
}
