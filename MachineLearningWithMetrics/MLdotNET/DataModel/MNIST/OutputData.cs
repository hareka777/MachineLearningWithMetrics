using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.MNIST
{
    public class OutputData
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
