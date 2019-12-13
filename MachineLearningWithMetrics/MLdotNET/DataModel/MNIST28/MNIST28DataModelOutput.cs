using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.MNIST28
{
    /*
    * Data model class for multi classification
    * Network output class
    */
    public class MNIST28DataModelOutput
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
