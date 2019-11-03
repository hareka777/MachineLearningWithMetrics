using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.MNIST
{
    public class InputData
    {
        [ColumnName("PixelValues")]
        [VectorType(64)]
        public float[] PixelValues;

        [LoadColumn(64)]
        public float Number;
    }
}
