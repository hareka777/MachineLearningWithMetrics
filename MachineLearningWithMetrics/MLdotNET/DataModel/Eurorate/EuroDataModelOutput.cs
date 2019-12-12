using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.Eurorate
{
    /*
     * Data model class for regression
     * Network output class
     */
    public class EuroDataModelOutput
    {
        [ColumnName("PredictedLabel")]
        public float Prediction;

        [ColumnName("Probability")]
        public float Probability;

        [ColumnName("Score")]
        public float Score;
    }
}
